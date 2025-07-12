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
                 speaking_to: Optional[str] = None):
        self.id = id
        self.content = content
        self.speaker = speaker
        self.timestamp = timestamp
        self.artifacts = artifacts
        self.speaking_to = speaking_to

    def to_prompt(self, **kwargs) -> str:
        artifacts_section = "\n".join([f"<li id=\"{a.id}\" type=\"{a.arch_type}\">" + a.to_prompt() + "</li>"
                                       for a in self.artifacts])

        speaking_to_section = f"<SpeakingTo>{self.speaking_to}</SpeakingTo>\n" if self.speaking_to else ""

        return (f"<Message id=\"{self.id}\" timestamp=\"{self.timestamp}\">\n"
                f"<Speaker>{self.speaker}</Speaker>\n"
                f"{speaking_to_section}"
                f"<Artifacts>\n"
                f"{artifacts_section}\n"
                f"</Artifacts>\n"
                f"<Content>{self.content}</Content>\n"
                f"</Message>")

    @staticmethod
    def make(content, speaker, artifacts=None, speaking_to=None):
        return Message(str(uuid.uuid4()), content, speaker, str(datetime.datetime.now()),
                       artifacts if artifacts else [], speaking_to)

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

        # Extract content
        content_match = re.search(r'<Content>(.*?)</Content>', response_text, re.DOTALL)
        content = content_match.group(1).strip() if content_match else ""

        # For now, artifacts parsing is simplified - you'd need to implement based on your Artifact classes
        artifacts = []

        return Message(msg_id, content, speaker, timestamp, artifacts, speaking_to)

    def __str__(self):
        speaking_to_str = f", speaking_to=\"{self.speaking_to}\"" if self.speaking_to else ""
        return (f"Message("
                f"speaker=\"{self.speaker}\"{speaking_to_str}, "
                f"timestamp=\"{self.timestamp}\", "
                f"content=\"{self.content}\", "
                f"artifacts=[{','.join([str(a.id) for a in self.artifacts])}]"
                f")")


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, speaker: str, messages: List[Message], stop_sequences: List[str] = None) -> Message:
        """
        Generate a response and return it as a complete Message object.

        Args:
            speaker: The speaker generating the response
            messages: List of previous messages
            stop_sequences: Optional stop sequences

        Returns:
            Complete Message object with the response
        """
        raise NotImplementedError("You must implement this method")