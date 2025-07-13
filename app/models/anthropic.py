import traceback

import anthropic
import uuid
import os
import datetime
from typing import List, Optional
from .base import BaseModel, Message  # Assuming your base classes are in a separate module


class AnthropicLLM(BaseModel):
    """
    Implementation of BaseModel for Anthropic's Claude models with whisper support.
    """

    def __init__(self,
                 api_key: str,
                 model: str = "claude-sonnet-4-20250514",
                 max_tokens: int = 4096,
                 temperature: float = 0.7):
        """
        Initialize the Anthropic LLM.

        Args:
            api_key: Your Anthropic API key
            model: Model name (default: claude-sonnet-4-20250514)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
        """
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _filter_messages_for_speaker(self, messages: List[Message], speaker: str) -> List[Message]:
        """
        Filter messages based on whisper visibility rules.

        Args:
            messages: List of all messages
            speaker: The speaker who will see these messages

        Returns:
            List of messages that the speaker can see
        """
        return [msg for msg in messages if msg.can_be_seen_by(speaker)]

    def _format_messages_for_anthropic(self, messages: List[Message], current_speaker: str) -> List[dict]:
        """
        Convert Message objects to Anthropic API format.

        Args:
            messages: List of Message objects
            current_speaker: The speaker who is currently generating a response

        Returns:
            List of dictionaries formatted for Anthropic API
        """
        formatted_messages = []

        for i, msg in enumerate(messages):
            # Messages from the current speaker become "assistant" messages
            # All other messages become "user" messages
            role = "assistant" if msg.speaker == current_speaker else "user"

            # Use the full to_prompt method for all messages
            content = msg.to_prompt()

            formatted_messages.append({
                "role": role,
                "content": content
            })

        # Handle the last message special case
        if messages and messages[-1].speaker == current_speaker:
            # Last message is from current speaker - extract scaffolding for continuation
            scaffolding_content = self._extract_scaffolding(messages[-1])
            formatted_messages[-1]["content"] = scaffolding_content
        else:
            # Last message is NOT from current speaker OR no messages yet - add temp scaffolding for new message
            temp_scaffolding = self._create_temp_scaffolding(current_speaker)
            formatted_messages.append({
                "role": "assistant",
                "content": temp_scaffolding
            })

        return formatted_messages

    def _ensure_alternating_roles(self, formatted_messages: List[dict]) -> List[dict]:
        """
        Ensure messages alternate between user and assistant roles as required by Anthropic API.

        Args:
            formatted_messages: List of formatted messages

        Returns:
            List of messages with alternating roles
        """
        if not formatted_messages:
            return formatted_messages

        # Anthropic API must start with user message
        if formatted_messages[0]["role"] != "user":
            # Insert a dummy user message at the start
            formatted_messages.insert(0, {
                "role": "user",
                "content": "Please begin the discussion."
            })

        # Ensure alternating pattern
        result = []
        for i, msg in enumerate(formatted_messages):
            result.append(msg)

            # Check if next message has same role
            if i < len(formatted_messages) - 1:
                next_msg = formatted_messages[i + 1]
                if msg["role"] == next_msg["role"]:
                    # Insert opposite role message
                    if msg["role"] == "user":
                        result.append({
                            "role": "assistant",
                            "content": "I understand. Please continue."
                        })
                    else:
                        result.append({
                            "role": "user",
                            "content": "Please continue with your assessment."
                        })

        return result

    def _create_temp_scaffolding(self, speaker: str) -> str:
        """
        Create temporary scaffolding for a new message from the current speaker.

        Args:
            speaker: The speaker who will be generating the new message
            speaking_to: Optional target of the message
            is_whisper: Whether this is a whisper message

        Returns:
            Temporary scaffolding structure with speaker and empty artifacts
        """
        temp_id = str(uuid.uuid4())
        temp_timestamp = str(datetime.datetime.now())

        return (f"<Message id=\"{temp_id}\" timestamp=\"{temp_timestamp}\">\n"
                f"<Speaker>{speaker}</Speaker>\n"
                f"<Artifacts>\n"
                f"</Artifacts>\n"
                f"<SpeakingTo>")

    def _extract_scaffolding(self, message: Message) -> str:
        """
        Extract the scaffolding from a message for continuation.
        If scaffolding exists, return everything up to and including <Content> with existing content.
        If no scaffolding, just return the message content as-is.

        Args:
            message: The message to extract scaffolding from

        Returns:
            The scaffolding portion for continuation, or plain content if no scaffolding
        """
        full_prompt = message.to_prompt()

        # Check if this looks like a structured message with scaffolding
        if '<Message' in full_prompt and '<Content>' in full_prompt:
            # Has scaffolding - extract everything up to and including existing content
            content_start = full_prompt.find('<Content>') + len('<Content>')
            content_end = full_prompt.find('</Content>')

            if content_start != -1 and content_end != -1:
                # Extract scaffolding + existing content
                scaffolding_part = full_prompt[:content_start]
                existing_content = full_prompt[content_start:content_end]
                return scaffolding_part + existing_content
            else:
                # Malformed, just return as-is
                return full_prompt
        else:
            # No scaffolding - just return the plain content
            return message.content

    def _extract_system_message(self, messages: List[Message]) -> tuple[Optional[str], List[Message]]:
        """
        Extract system message if the first message is from 'system' speaker.

        Args:
            messages: List of Message objects

        Returns:
            Tuple of (system_message, remaining_messages)
        """
        if messages and messages[0].speaker.lower() == "system":
            return messages[0].content, messages[1:]
        return None, messages

    def __call__(self,
                 speaker: str,
                 messages: List[Message],
                 stop_sequences: List[str] = None) -> Message:
        """
        Generate a response using Anthropic's API.
        """
        try:
            messages = self.prepare(speaker, messages)

            # Filter messages based on whisper visibility
            filtered_messages = self._filter_messages_for_speaker(messages, speaker)

            # Extract system message if present
            system_message, user_messages = self._extract_system_message(filtered_messages)

            # Format messages for Anthropic API
            formatted_messages = self._format_messages_for_anthropic(user_messages, speaker)

            # Ensure alternating roles
            formatted_messages = self._ensure_alternating_roles(formatted_messages)

            # If we're creating new scaffolding and have a speaking_to target, update it
            if not formatted_messages:
                formatted_messages = []
            if formatted_messages[-1]["role"] != "assistant":
                formatted_messages.append({"role": "assistant", "content": self._create_temp_scaffolding(speaker)})

            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": formatted_messages
            }

            if system_message:
                api_params["system"] = system_message
            if stop_sequences:
                api_params["stop_sequences"] = stop_sequences

            # Make the API call
            response = self.client.messages.create(**api_params)
            response_text = formatted_messages[-1]['content'] + response.content[0].text

            # Try to parse the response
            try:
                return Message.parse_from_response(response_text)
            except Exception as parse_error:
                print(f"⚠️ Failed to parse LLM response, retrying with forced scaffolding: {parse_error}")

                # FALLBACK: Completely redo the call with forced scaffolding
                recovery_scaffolding = self._create_recovery_scaffolding(speaker, "All", False)

                # Create new formatted messages with forced scaffolding as the assistant message
                recovery_formatted_messages = formatted_messages[:-1]  # Remove the last assistant message
                recovery_formatted_messages.append({
                    "role": "assistant",
                    "content": recovery_scaffolding
                })

                # Update API params with new messages
                recovery_api_params = api_params.copy()
                recovery_api_params["messages"] = recovery_formatted_messages

                # Second attempt with forced scaffolding
                recovery_response = self.client.messages.create(**recovery_api_params)
                recovery_text = recovery_scaffolding + recovery_response.content[0].text

                try:
                    return Message.parse_from_response(recovery_text)
                except Exception as recovery_parse_error:
                    print(f"❌ Recovery attempt also failed: {recovery_parse_error}")
                    # Last resort: create a basic message manually
                    return Message.make(
                        content=recovery_response.content[0].text,
                        speaker=speaker,
                        speaking_to="All",  # Speak to everyone as fallback
                        is_whisper=False
                    )

        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Anthropic API: {str(e)}")

    def _create_recovery_scaffolding(self, speaker: str, speaking_to: Optional[str] = None,
                                     is_whisper: bool = False) -> str:
        """
        Create complete scaffolding up to <Content> tag for recovery attempt.
        This forces the LLM to continue from the proper structure.
        """
        temp_id = str(uuid.uuid4())
        temp_timestamp = str(datetime.datetime.now())

        scaffolding = (f"<Message id=\"{temp_id}\" timestamp=\"{temp_timestamp}\">\n"
                       f"<Speaker>{speaker}</Speaker>\n")

        # Add optional fields - if not provided, speak to everyone (no whisper)
        if speaking_to:
            scaffolding += f"<SpeakingTo>{speaking_to}</SpeakingTo>\n"

        if is_whisper:
            scaffolding += f"<Whisper>true</Whisper>\n"

        scaffolding += (f"<Artifacts>\n"
                        f"</Artifacts>\n"
                        f"<Content>")

        return scaffolding

    def set_model(self, model: str):
        """Change the model being used."""
        self.model = model

    def set_temperature(self, temperature: float):
        """Change the temperature setting."""
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int):
        """Change the max tokens setting."""
        if max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        self.max_tokens = max_tokens
