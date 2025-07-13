import random
import time
import regex
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from models.anthropic import AnthropicLLM
from models.base import Message


class AgentType(Enum):
    PARTICIPANT = "PARTICIPANT"
    COORDINATOR = "COORDINATOR"


@dataclass
class Persona:
    """Represents an agent with their role and personality."""
    name: str
    title: str
    expertise: str
    personality: str
    speaking_style: str
    agent_type: AgentType = AgentType.PARTICIPANT


@dataclass
class Goal:
    """Represents a trackable goal in the conversation."""
    name: str
    description: str
    achieved: bool = False
    achievement_message: Optional[str] = None


@dataclass
class RejectionResult:
    """Result of a validity check."""
    is_valid: bool
    rejection_reason: Optional[str] = None

    @classmethod
    def valid(cls):
        return cls(is_valid=True)

    @classmethod
    def invalid(cls, reason: str):
        return cls(is_valid=False, rejection_reason=reason)


class ValidityChecker(ABC):
    """Abstract base class for message validity checks."""

    @abstractmethod
    def check(self, message: Message, config: Any, participant_state: Any) -> RejectionResult:
        """Check if message is valid."""
        pass


@dataclass
class AgentState:
    """Tracks the state of each agent."""
    persona: Persona
    has_withdrawn: bool = False
    message_count: int = 0
    custom_data: Dict[str, Any] = field(default_factory=dict)


class CoordinatorConfig:
    """Configuration for Coordinator behavior."""

    def __init__(self,
                 intervention_interval: int = 4,
                 escalation_thresholds: List[int] = None):
        self.intervention_interval = intervention_interval
        self.escalation_thresholds = escalation_thresholds or [20, 35, 50]


class AgentOrchestrator:
    """
    Universal system for orchestrating multi-agent conversations with validation.
    """

    def __init__(self,
                 api_key: str,
                 conversation_topic: str,
                 context_content: str,
                 coordinator_config: CoordinatorConfig = None,
                 validity_checkers: List[ValidityChecker] = None,
                 goals: List[Goal] = None):
        self.llm = AnthropicLLM(api_key=api_key, temperature=0.8)
        self.messages: List[Message] = []
        self.conversation_topic = conversation_topic
        self.context_content = context_content
        self.coordinator_config = coordinator_config or CoordinatorConfig()
        self.validity_checkers = validity_checkers or []
        self.goals = goals or []

        self.conversation_active = True
        self.message_count = 0
        self.max_messages = 100
        self.agents: Dict[str, AgentState] = {}

        # Coordinator tracking
        self.last_intervention = 0
        self.rejection_count = 0

        # Create default coordinator persona
        self.coordinator_persona = Persona(
            name="Coordinator",
            title="Conversation Coordinator",
            expertise="Discussion management, goal tracking, process facilitation",
            personality="Efficient and diplomatic, focused on achieving objectives",
            speaking_style="Clear and direct, uses structured language and progress tracking",
            agent_type=AgentType.COORDINATOR
        )

    def setup_agents(self, personas: List[Persona]):
        """Initialize agents from persona list."""
        for persona in personas:
            self.agents[persona.name.lower()] = AgentState(persona=persona)

        # Add Coordinator as special agent
        self.agents["coordinator"] = AgentState(persona=self.coordinator_persona)

    def validate_message(self, message: Message, agent_name: str) -> RejectionResult:
        """Run all validity checks on a message."""
        agent_state = self.agents[agent_name.lower()]

        # Skip validation for Coordinator
        if agent_state.persona.agent_type == AgentType.COORDINATOR:
            return RejectionResult.valid()

        for checker in self.validity_checkers:
            result = checker.check(message, self.get_validation_config(), agent_state)
            if not result.is_valid:
                return result

        return RejectionResult.valid()

    def get_validation_config(self) -> Any:
        """Override this to provide configuration for validators."""
        return None

    def create_rejection_response(self, rejected_agent: str, rejection_reason: str) -> Message:
        """Create Coordinator response for message rejection."""
        agent_name = self.agents[rejected_agent.lower()].persona.name

        content = f"""❌ FORMAT ERROR - {agent_name}

Your message was rejected:
{rejection_reason}

Please resubmit following the proper format."""

        return Message.make(
            content=content,
            speaker="coordinator",
            speaking_to=agent_name
        )

    def parse_response_fields(self, message: Message) -> Tuple[Dict[str, Any], List[str]]:
        """Parse custom fields and achieved goals from message."""
        full_response = message.to_prompt()

        # Extract achieved goals using regex
        achieved_goals = []
        goal_matches = regex.findall(r"<GoalAchieved>(?:\s|\n)*([^<]+?)(?:\s|\n)*</GoalAchieved>", full_response)
        achieved_goals.extend([goal.strip() for goal in goal_matches])

        # Override this for custom field parsing
        custom_fields = self.parse_custom_fields(full_response)

        return custom_fields, achieved_goals

    def parse_custom_fields(self, full_response: str) -> Dict[str, Any]:
        """Override this to parse domain-specific fields."""
        return {}

    def update_achieved_goals(self, achieved_goal_names: List[str], message_content: str):
        """Update goals that have been achieved."""
        for goal_name in achieved_goal_names:
            for goal in self.goals:
                if goal.name.lower() == goal_name.lower() and not goal.achieved:
                    goal.achieved = True
                    goal.achievement_message = message_content
                    print(f"🎯 GOAL ACHIEVED: {goal.name}")

    def get_active_agents(self) -> List[str]:
        """Get list of agents who haven't withdrawn (excludes Coordinator)."""
        return [name for name, state in self.agents.items()
                if not state.has_withdrawn and state.persona.agent_type == AgentType.PARTICIPANT]

    def should_coordinator_intervene(self) -> bool:
        """Determine if Coordinator should intervene."""
        return self.message_count - self.last_intervention >= self.coordinator_config.intervention_interval

    def get_coordinator_urgency_level(self) -> str:
        """Determine the urgency level for Coordinator interventions."""
        thresholds = self.coordinator_config.escalation_thresholds

        if self.message_count >= thresholds[2]:
            return "critical"
        elif self.message_count >= thresholds[1]:
            return "urgent"
        elif self.message_count >= thresholds[0]:
            return "elevated"
        else:
            return "normal"

    def get_coordinator_message_content(self, urgency_level: str, rejection_context: str = None) -> str:
        """Generate content for Coordinator interventions. Override for custom behavior."""
        if rejection_context:
            return f"""🔴 FORMAT CORRECTION NEEDED

{rejection_context}

Please ensure proper formatting in responses."""

        if urgency_level == "critical":
            return f"""⚠️ CRITICAL: Conversation must conclude soon. Message {self.message_count}"""
        elif urgency_level == "urgent":
            return f"""🔴 URGENT: We need to wrap up. Message {self.message_count}"""
        elif urgency_level == "elevated":
            return f"""🟡 ATTENTION: Progress check at message {self.message_count}"""
        else:
            return f"""📋 Process reminder - Message {self.message_count}"""

    def generate_next_speaker(self, last_speaker: str) -> Optional[str]:
        """Determine who should speak next."""
        # Check if Coordinator should intervene
        if self.should_coordinator_intervene() and last_speaker != "coordinator":
            return "coordinator"

        active_agents = self.get_active_agents()
        if not active_agents:
            return None

        # Remove last speaker to avoid back-and-forth
        if last_speaker and last_speaker.lower() in active_agents:
            active_agents.remove(last_speaker.lower())

        if not active_agents:
            active_agents = self.get_active_agents()

        # Weight by message count (less = more likely to speak)
        weights = []
        for agent_name in active_agents:
            agent_state = self.agents[agent_name]
            weight = max(1, 10 - agent_state.message_count)
            weights.append(weight)

        return random.choices(active_agents, weights=weights)[0]

    def determine_speaking_to(self, speaker: str, last_speaker: str) -> Optional[str]:
        """Determine who the speaker might be addressing."""
        speaker_state = self.agents[speaker.lower()]

        if speaker_state.persona.agent_type == AgentType.COORDINATOR:
            return None  # Coordinator addresses the group

        if last_speaker and random.random() < 0.3:
            return last_speaker
        return None

    def all_agents_withdrawn(self) -> bool:
        """Check if all agents (excluding Coordinator) have withdrawn."""
        return all(state.has_withdrawn for name, state in self.agents.items()
                   if state.persona.agent_type == AgentType.PARTICIPANT)

    def get_shared_system_prompt(self) -> str:
        """Generate the shared system prompt for all agents. Override for customization."""
        raise NotImplementedError

    def get_agent_system_prompt(self, agent_name: str) -> str:
        """Generate complete system prompt for an agent. Override for customization."""
        raise NotImplementedError

    def add_system_message(self, speaker: str) -> List[Message]:
        """Add system prompt as first message for the speaker's context."""
        system_prompt = self.get_agent_system_prompt(speaker)
        system_msg = Message.make(
            content=system_prompt,
            speaker="system"
        )
        return [system_msg] + self.messages

    def print_message(self, message: Message, custom_fields: Dict[str, Any], achieved_goals: List[str]):
        """Print a message with custom information."""
        agent_state = self.agents[message.speaker.lower()]
        persona = agent_state.persona

        # Different formatting for Coordinator
        if persona.agent_type == AgentType.COORDINATOR:
            urgency = self.get_coordinator_urgency_level()
            urgency_indicator = {"normal": "📋", "elevated": "🟡", "urgent": "🔴", "critical": "⚠️"}[urgency]
            print(f"\n[{self.message_count}] {urgency_indicator} {persona.name.upper()} → [All]:")
            print(f"    {message.content}")
            if achieved_goals:
                print(f"    🎯 Goals marked achieved: {', '.join(achieved_goals)}")
            return

        speaking_to_str = f" → {message.speaking_to}" if message.speaking_to else " → [All]"

        print(f"\n[{self.message_count}] {persona.name.upper()} ({persona.title}){speaking_to_str}:")
        print(f"(is_whisper={message.is_whisper})")
        print(f"    {message.content}")

        # Print custom fields if any
        if custom_fields:
            for key, value in custom_fields.items():
                print(f"    {key}: {value}")

        print(f"    (timestamp: {message.timestamp})")

    def print_final_results(self):
        """Print final results. Override for custom output."""
        print("\n" + "=" * 70)
        print("📋 FINAL CONVERSATION RESULTS")
        print("=" * 70)

        # Goal achievements
        if self.goals:
            print("\n🎯 GOAL ACHIEVEMENTS:")
            for goal in self.goals:
                status = "✅ ACHIEVED" if goal.achieved else "❌ NOT ACHIEVED"
                print(f"   {goal.name}: {status}")
                if goal.achieved and goal.achievement_message:
                    print(f"      Context: {goal.achievement_message[:100]}...")

    def run_conversation(self):
        """Run the automated conversation."""
        print(f"🏛️  AGENT CONVERSATION STARTING")
        print(f"Topic: {self.conversation_topic}")
        print(
            f"Participants: {', '.join([state.persona.name for name, state in self.agents.items() if state.persona.agent_type == AgentType.PARTICIPANT])}")
        if self.goals:
            print(f"Goals: {', '.join([goal.name for goal in self.goals])}")
        print("=" * 70)

        # Start with a random participant
        current_speaker = random.choice([name for name, state in self.agents.items()
                                         if state.persona.agent_type == AgentType.PARTICIPANT])

        while self.conversation_active and self.message_count < self.max_messages:
            try:
                if self.all_agents_withdrawn():
                    print(f"\n🏁 All participants have withdrawn!")
                    self.conversation_active = False
                    break

                # Get messages with system prompt
                messages_with_system = self.add_system_message(current_speaker)

                # Determine speaking target
                last_speaker = self.messages[-1].speaker if self.messages else None
                speaking_to = self.determine_speaking_to(current_speaker, last_speaker) if last_speaker else None

                agent_state = self.agents[current_speaker]
                print(f"\n🤔 {agent_state.persona.name} is considering their response...")

                if current_speaker == "coordinator":
                    # Coordinator message
                    urgency_level = self.get_coordinator_urgency_level()
                    coordinator_content = self.get_coordinator_message_content(urgency_level)
                    response_msg = Message.make(
                        content=coordinator_content,
                        speaker=current_speaker,
                        speaking_to=speaking_to
                    )
                else:
                    # Generate response for regular participants
                    response_msg = self.llm(
                        speaker=current_speaker,
                        messages=messages_with_system,
                        speaking_to=speaking_to
                    )

                # Validate message for regular participants
                if agent_state.persona.agent_type == AgentType.PARTICIPANT:
                    validation_result = self.validate_message(response_msg, current_speaker)
                    if not validation_result.is_valid:
                        print(f"❌ Message rejected: {validation_result.rejection_reason}")

                        # Coordinator intervenes with rejection response
                        rejection_response = self.create_rejection_response(current_speaker,
                                                                            validation_result.rejection_reason)
                        self.messages.append(rejection_response)
                        self.message_count += 1
                        self.rejection_count += 1

                        # Print rejection message
                        print(
                            f"\n[{self.message_count}] ❌ Coordinator → {self.agents[current_speaker.lower()].persona.name}:")
                        print(f"    {rejection_response.content}")

                        # Give same participant another chance
                        continue

                # Parse response fields
                custom_fields, achieved_goals = self.parse_response_fields(response_msg)

                # Update agent state
                agent_state.message_count += 1
                self.update_agent_state(current_speaker, custom_fields)

                # Update achieved goals
                if achieved_goals:
                    self.update_achieved_goals(achieved_goals, response_msg.content)

                # Update Coordinator tracking
                if current_speaker == "coordinator":
                    self.last_intervention = self.message_count

                # Add to conversation
                self.messages.append(response_msg)
                self.message_count += 1

                # Print the message
                self.print_message(response_msg, custom_fields, achieved_goals)

                # Determine next speaker
                next_speaker = self.generate_next_speaker(current_speaker)
                if not next_speaker:
                    print(f"\n🏁 No active participants remaining!")
                    self.conversation_active = False
                    break

                current_speaker = next_speaker
                time.sleep(0.5)

            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print(f"Current speaker: {current_speaker}")
                break

        # Print final results
        self.print_final_results()

        print(f"\n📈 CONVERSATION STATISTICS:")
        print(f"   Total messages: {self.message_count}")
        print(f"   Message rejections: {self.rejection_count}")
        print(f"   Active participants: {len(self.get_active_agents())}")
        print(f"   Coordinator interventions: {self.agents['coordinator'].message_count}")

        return {
            'goals_achieved': [goal.name for goal in self.goals if goal.achieved],
            'message_count': self.message_count,
            'rejections': self.rejection_count
        }

    def update_agent_state(self, agent_name: str, custom_fields: Dict[str, Any]):
        """Update agent state with custom fields. Override for domain-specific behavior."""
        agent_state = self.agents[agent_name.lower()]
        agent_state.custom_data.update(custom_fields)
