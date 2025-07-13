import anthropic
import uuid
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

    def _create_temp_scaffolding(self, speaker: str, speaking_to: str = None, is_whisper: bool = False) -> str:
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

        speaking_to_section = f"<SpeakingTo>{speaking_to}</SpeakingTo>\n" if speaking_to else ""
        whisper_section = f"<Whisper>true</Whisper>\n" if is_whisper else ""

        return (f"<Message id=\"{temp_id}\" timestamp=\"{temp_timestamp}\">\n"
                f"<Speaker>{speaker}</Speaker>\n"
                f"{speaking_to_section}"
                f"{whisper_section}"
                f"<Artifacts>\n"
                f"</Artifacts>\n"
                f"<Content>")

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
                 stop_sequences: List[str] = None,
                 speaking_to: str = None,
                 is_whisper: bool = False) -> Message:
        """
        Generate a response using Anthropic's API.

        Args:
            speaker: The speaker who is generating the response (their messages become "assistant" role)
            messages: List of Message objects representing the conversation
            stop_sequences: Optional list of sequences to stop generation
            speaking_to: Optional target speaker for the response
            is_whisper: Whether this response should be a whisper

        Returns:
            Complete Message object with the generated response
        """
        try:
            messages = self.prepare(speaker, messages, speaking_to)

            # DEBUG: Print the system message to see if scaffolding is there
            for msg in messages:
                if msg.speaker == "system":
                    print(f"🔍 SYSTEM MESSAGE LENGTH: {len(msg.content)}")
                    print(f"🔍 CONTAINS SCAFFOLDING: {'<Verdict>' in msg.content}")
                    break
            # Filter messages based on whisper visibility - the current speaker can only see
            # messages that they are allowed to see (public messages + whispers directed to them)
            filtered_messages = self._filter_messages_for_speaker(messages, speaker)

            # Extract system message if present
            system_message, user_messages = self._extract_system_message(filtered_messages)

            # Format messages for Anthropic API
            formatted_messages = self._format_messages_for_anthropic(user_messages, speaker)

            # Anthropic API requires alternating user/assistant messages
            # Ensure we don't have consecutive messages with the same role
            formatted_messages = self._ensure_alternating_roles(formatted_messages)

            # If we're creating new scaffolding and have a speaking_to target, update it
            if (not user_messages or user_messages[-1].speaker != speaker) and (speaking_to or is_whisper):
                if formatted_messages and formatted_messages[-1]["role"] == "assistant":
                    # Update the temp scaffolding to include speaking_to and whisper status
                    temp_scaffolding = self._create_temp_scaffolding(speaker, speaking_to, is_whisper)
                    formatted_messages[-1]["content"] = temp_scaffolding

            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": formatted_messages
            }

            # Add system message if present
            if system_message:
                api_params["system"] = system_message

            # Add stop sequences if provided
            if stop_sequences:
                api_params["stop_sequences"] = stop_sequences

            # Make the API call
            response = self.client.messages.create(**api_params)

            # Get the complete response text
            response_text = response.content[0].text

            # If the response doesn't contain scaffolding (shouldn't happen with our setup),
            # we need to construct it
            if not response_text.startswith("<Message"):
                # Create complete scaffolding with the response
                temp_id = str(uuid.uuid4())
                temp_timestamp = str(datetime.datetime.now())
                speaking_to_section = f"<SpeakingTo>{speaking_to}</SpeakingTo>\n" if speaking_to else ""
                whisper_section = f"<Whisper>true</Whisper>\n" if is_whisper else ""

                complete_response = (f"<Message id=\"{temp_id}\" timestamp=\"{temp_timestamp}\">\n"
                                     f"<Speaker>{speaker}</Speaker>\n"
                                     f"{speaking_to_section}"
                                     f"{whisper_section}"
                                     f"<Artifacts>\n"
                                     f"</Artifacts>\n"
                                     f"<Content>{response_text}</Content>\n"
                                     f"</Message>")
            else:
                complete_response = response_text
                # Ensure it has proper closing tags
                if not complete_response.endswith("</Message>"):
                    if not complete_response.endswith("</Content>"):
                        complete_response += "</Content>"
                    complete_response += "\n</Message>"

            # Parse the complete response into a Message object
            return Message.parse_from_response(complete_response)

        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Anthropic API: {str(e)}")

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
