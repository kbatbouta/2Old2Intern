from typing import Callable
from agents.agent_system import DebateWatcher, AgentOrchestratorAPI


class ScheduledMessage(DebateWatcher):
    """
    A tick-based plugin that injects a message when specified conditions are met.

    ScheduledMessage acts as a one-shot intervention that monitors conversation state
    on every turn (tick) and injects a predefined message when its trigger condition
    becomes true. After triggering once, it becomes inactive for the remainder of
    the conversation.

    Args:
        message_content: The message text to inject when triggered
        trigger: A callable that receives (current_speaker, orchestrator_api) and
                returns True when the message should be injected
        speaker: The speaker name for the injected message (default: "coordinator")

    Example:
        >>> def after_10_messages(speaker, api):
        ...     return len(list(api.messages())) >= 10
        >>>
        >>> reminder = ScheduledMessage(
        ...     "Time to wrap up the discussion!",
        ...     after_10_messages
        ... )
        >>> orchestrator.watchers.append(reminder)

    Note:
        The trigger function is evaluated on every conversation turn. For expensive
        operations, consider caching or state tracking within the trigger itself.
    """
    def __init__(self, message_content: str, trigger: Callable[[str, AgentOrchestratorAPI], bool], speaker: str = "coordinator"):
        self.speaker = speaker
        self.message_content = message_content
        self.trigger = trigger
        self._finished = False

    def __call__(self, current_speaker, orchestrator_api: AgentOrchestratorAPI):
        if self._finished:
            return
        if self.trigger(current_speaker, orchestrator_api):
            self._finished = True
            orchestrator_api.inject_message(self.message_content, self.speaker)