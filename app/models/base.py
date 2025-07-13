import abc
import datetime
import uuid
from typing import List, Optional


class Artifact(abc.ABC):
    def __init__(self, id: str, art_type: str):
        self.id = id
        self.arch_type = art_type

    @abc.abstractmethod
    def to_prompt(self, **kwargs) -> str:
        raise NotImplementedError


class Message(object):
    def __init__(self, id: str, content: str, speaker: str, timestamp: str, artifacts: List[Artifact],
                 speaking_to: Optional[str] = None, is_whisper: bool = False):
        self.id = id
        self.content = content
        self.speaker = speaker
        self.timestamp = timestamp
        self.artifacts = artifacts
        self.speaking_to = speaking_to
        self.is_whisper = is_whisper

    def to_prompt(self, **kwargs) -> str:
        artifacts_section = "\n".join([f"<li id=\"{a.id}\" type=\"{a.arch_type}\">" + a.to_prompt() + "</li>"
                                       for a in self.artifacts])

        speaking_to_section = f"<SpeakingTo>{self.speaking_to}</SpeakingTo>\n" if (
            self.speaking_to) else "<SpeakingTo>All</SpeakingTo>"
        whisper_section = f"<Whisper>true</Whisper>\n" if self.is_whisper else "<Whisper>False</Whisper>"

        return (f"<Message id=\"{self.id}\" timestamp=\"{self.timestamp}\">\n"
                f"<Speaker>{self.speaker}</Speaker>\n"
                f"{speaking_to_section}"
                f"{whisper_section}"
                f"<Artifacts>\n"
                f"{artifacts_section}\n"
                f"</Artifacts>\n"
                f"<Content>{self.content}</Content>\n"
                f"</Message>")

    @staticmethod
    def make(content, speaker, artifacts=None, speaking_to=None, is_whisper=False):
        return Message(str(uuid.uuid4()), content, speaker, str(datetime.datetime.now()),
                       artifacts if artifacts else [], speaking_to, is_whisper)

    @staticmethod
    def parse_from_response(response_text: str) -> 'Message':
        """
        Parse a Message object from LLM response text that contains scaffolding.

        Args:
            response_text: The complete response from the LLM including scaffolding

        Returns:
            Message object parsed from the response
        """
        import re

        # Extract message ID
        id_match = re.search(r'<Message id="([^"]*)"', response_text)
        msg_id = id_match.group(1) if id_match else str(uuid.uuid4())

        # Extract timestamp
        timestamp_match = re.search(r'timestamp="([^"]*)"', response_text)
        timestamp = timestamp_match.group(1) if timestamp_match else str(datetime.datetime.now())

        # Extract speaker
        speaker_match = re.search(r'<Speaker>([^<]*)</Speaker>', response_text)
        speaker = speaker_match.group(1) if speaker_match else "unknown"

        # Extract speaking_to
        speaking_to_match = re.search(r'<SpeakingTo>([^<]*)</SpeakingTo>', response_text)
        speaking_to = speaking_to_match.group(1) if speaking_to_match else None

        # Extract whisper flag
        whisper_match = re.search(r'<Whisper>([^<]*)</Whisper>', response_text)
        is_whisper = whisper_match.group(1).lower() == 'true' if whisper_match else False

        # Extract content
        content_match = re.search(r'<Content>(.*?)</Content>', response_text, re.DOTALL)
        content = content_match.group(1).strip() if content_match else ""

        # For now, artifacts parsing is simplified - you'd need to implement based on your Artifact classes
        artifacts = []

        return Message(msg_id, content, speaker, timestamp, artifacts, speaking_to, is_whisper)

    def can_be_seen_by(self, speaker: str) -> bool:
        """
        Determine if a message can be seen by a specific speaker.

        Args:
            speaker: The speaker to check visibility for

        Returns:
            True if the speaker can see this message, False otherwise
        """
        if not self.is_whisper:
            # Non-whisper messages can be seen by everyone
            return True

        # Whisper messages can only be seen by the speaker and the target
        return speaker == self.speaker or speaker == self.speaking_to

    def __str__(self):
        speaking_to_str = f", speaking_to=\"{self.speaking_to}\"" if self.speaking_to else ""
        whisper_str = f", whisper={self.is_whisper}" if self.is_whisper else ""
        return (f"Message("
                f"speaker=\"{self.speaker}\"{speaking_to_str}{whisper_str}, "
                f"timestamp=\"{self.timestamp}\", "
                f"content=\"{self.content}\", "
                f"artifacts=[{','.join([str(a.id) for a in self.artifacts])}]"
                f")")


class BaseModel(abc.ABC):

    def get_scaffolding_examples(self, speaker: str) -> str:
        """
        Get scaffolding examples for the given speaker. Override this method
        to provide context-specific scaffolding examples.

        Args:
            speaker: The speaker who will be generating the response

        Returns:
            String containing scaffolding examples to append to system message
        """
        return """
RESPONSE SCAFFOLDING EXAMPLES:

Example 1 - Regular message (visible to everyone):
<Message id="example-123" timestamp="2024-01-15T10:30:00">
<Speaker>John</Speaker>
<Artifacts>
</Artifacts>
<Content>I think we should consider the technical implications here.</Content>
</Message>

Example 2 - Message with speaking target (visible to everyone):
<Message id="example-456" timestamp="2024-01-15T10:31:00">
<Speaker>Sarah</Speaker>
<SpeakingTo>John</SpeakingTo>
<Artifacts>
</Artifacts>
<Content>John, what's your take on the performance metrics?</Content>
</Message>

Example 3 - Whisper message (ONLY visible to speaker and target):
<Message id="example-789" timestamp="2024-01-15T10:32:00">
<Speaker>Mike</Speaker>
<SpeakingTo>Sarah</SpeakingTo>
<Artifacts>
</Artifacts>
<Content>I don't think John's numbers are accurate.</Content>
<Whisper>true</Whisper>
</Message>

IMPORTANT: Whisper messages with <Whisper>true</Whisper> are PRIVATE and can only be seen by:
- The speaker (person sending the whisper)
- The target specified in <SpeakingTo>

All other participants will NOT see whisper messages. Use whispers for private communications.

Remember you can whisper to others, and you should - but not too often.

Always use this exact scaffolding format in your responses.
"""

    def prepare(self, speaker: str, messages: List[Message]) -> List[Message]:
        """
        Prepare messages for LLM call by adding scaffolding examples to system message.
        Creates a system message if one doesn't exist. Non-destructive operation.

        Args:
            speaker: The speaker generating the response
            messages: Original list of messages
            speaking_to: Optional target for the response

        Returns:
            New list of messages with scaffolding examples added to system message
        """
        # Create a copy of the messages to avoid modifying the original
        prepared_messages = messages.copy()

        # Check if system message exists
        system_message_index = None
        for i, msg in enumerate(prepared_messages):
            if msg.speaker == "system":
                system_message_index = i
                break

        # Get scaffolding examples
        scaffolding_examples = self.get_scaffolding_examples(speaker)

        if system_message_index is not None:
            # System message exists - append scaffolding to its content
            existing_system_msg = prepared_messages[system_message_index]
            enhanced_content = existing_system_msg.content + "\n\n" + scaffolding_examples

            # Create new system message with enhanced content
            enhanced_system_msg = Message(
                id=existing_system_msg.id,
                content=enhanced_content,
                speaker="system",
                timestamp=existing_system_msg.timestamp,
                artifacts=existing_system_msg.artifacts,
                speaking_to=existing_system_msg.speaking_to,
                is_whisper=existing_system_msg.is_whisper
            )

            # Replace the system message
            prepared_messages[system_message_index] = enhanced_system_msg
        else:
            # No system message exists - create one with scaffolding
            system_msg = Message.make(
                content=f"You are participating in a structured conversation.\n\n{scaffolding_examples}",
                speaker="system"
            )

            # Insert at the beginning
            prepared_messages.insert(0, system_msg)

        return prepared_messages

    @abc.abstractmethod
    def __call__(self, speaker: str, messages: List[Message], stop_sequences: List[str] = None) -> Message:
        """
        Generate a response and return it as a complete Message object.
        Should call prepare() internally before making the LLM call.

        Args:
            speaker: The speaker generating the response
            messages: List of previous messages
            speaking_to: Optional target for the response
            stop_sequences: Optional stop sequences

        Returns:
            Complete Message object with the response
        """
        raise NotImplementedError("You must implement this method")