import regex
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from agents.agent_system import (
    AgentOrchestrator, ValidityChecker, RejectionResult,
    Persona, Goal, AgentType, CoordinatorConfig, DebateWatcher
)
from models.base import BaseModel


@dataclass
class VerdictConfig:
    """Configuration for verdict options in the debate."""
    verdict_options: List[str]
    verdict_descriptions: Dict[str, str] = field(default_factory=dict)

    def get_verdict_prompt(self) -> str:
        """Generate prompt text for verdict options."""
        if self.verdict_descriptions:
            options = []
            for option in self.verdict_options:
                desc = self.verdict_descriptions.get(option, "")
                options.append(f"{option}: {desc}" if desc else option)
            return " | ".join(options)
        return " | ".join(self.verdict_options)


class VerdictValidityChecker(ValidityChecker):
    """Checks if verdict is valid when provided."""

    def check(self, message, config: VerdictConfig, participant_state) -> RejectionResult:
        full_response = message.to_prompt()

        # Check if verdict tag is present
        verdict_match = regex.search(r"<Verdict>(?:\s|\n)*(.*?)(?:\s|\n)*</Verdict>", full_response, regex.DOTALL)
        if not verdict_match:
            return RejectionResult.valid()  # No verdict tag is fine

        verdict_content = verdict_match.group(1).strip()

        # Empty verdict is okay
        if not verdict_content:
            return RejectionResult.valid()

        # Check if verdict matches allowed options
        for option in config.verdict_options:
            if verdict_content == option:
                return RejectionResult.valid()

        valid_options = " | ".join(config.verdict_options)
        return RejectionResult.invalid(
            f"Invalid verdict '{verdict_content}'. Must be one of: {valid_options}"
        )


class VerdictReasoningChecker(ValidityChecker):
    """Checks if verdict reasoning is provided when verdict is given."""

    def check(self, message, config: VerdictConfig, participant_state) -> RejectionResult:
        full_response = message.to_prompt()

        # Check if verdict is present and not empty
        verdict_match = regex.search(r"<Verdict>(?:\s|\n)*(.*?)(?:\s|\n)*</Verdict>", full_response, regex.DOTALL)
        if not verdict_match or not verdict_match.group(1).strip():
            return RejectionResult.valid()  # No verdict, so reasoning not required

        # Check if reasoning tag is present
        reasoning_match = regex.search(r"<VerdictReasoning>(?:\s|\n)*(.*?)(?:\s|\n)*</VerdictReasoning>", full_response,
                                       regex.DOTALL)
        if not reasoning_match or not reasoning_match.group(1).strip():
            return RejectionResult.invalid(
                "Verdict reasoning must be provided when a verdict is given. Please include <VerdictReasoning>your explanation</VerdictReasoning>"
            )

        return RejectionResult.valid()


class WithdrawalValidityChecker(ValidityChecker):
    """Checks if withdrawal is valid - requires a verdict first."""

    def check(self, message, config: VerdictConfig, participant_state) -> RejectionResult:
        full_response = message.to_prompt()

        # Check if trying to withdraw
        withdrawn_match = regex.search(r"<Withdrawn>(?:\s|\n)*(.*?)(?:\s|\n)*</Withdrawn>", full_response, regex.DOTALL)
        if not withdrawn_match:
            return RejectionResult.valid()  # No withdrawal attempt

        withdrawn_content = withdrawn_match.group(1).strip().lower()
        if withdrawn_content != "true":
            return RejectionResult.valid()  # Not withdrawing

        # Check if they have a current verdict
        current_verdict = participant_state.custom_data.get('verdict')

        # Check if they're providing a verdict in this message
        verdict_in_message = None
        for option in config.verdict_options:
            if regex.findall(rf"<Verdict>(?:\s|\n)*{regex.escape(option)}(?:\s|\n)*</Verdict>", full_response):
                verdict_in_message = option
                break

        # Allow withdrawal if they have a previous verdict OR providing one now
        if current_verdict or verdict_in_message:
            return RejectionResult.valid()

        return RejectionResult.invalid(
            "Cannot withdraw without providing a verdict. Please include <Verdict>your_choice</Verdict> and <VerdictReasoning>your explanation</VerdictReasoning> before withdrawing."
        )


class WithdrawnFormatChecker(ValidityChecker):
    """Checks if withdrawn tag is properly formatted when present."""

    def check(self, message, config: VerdictConfig, participant_state) -> RejectionResult:
        full_response = message.to_prompt()

        # Check if withdrawn tag is present
        withdrawn_match = regex.search(r"<Withdrawn>(?:\s|\n)*(.*?)(?:\s|\n)*</Withdrawn>", full_response, regex.DOTALL)
        if not withdrawn_match:
            return RejectionResult.valid()  # No withdrawn tag is fine

        withdrawn_content = withdrawn_match.group(1).strip().lower()
        if withdrawn_content not in ["true", "false"]:
            return RejectionResult.invalid(
                f"Invalid withdrawn value '{withdrawn_content}'. Must be 'true' or 'false'"
            )

        return RejectionResult.valid()


class DebateTimeKeeperConfig(CoordinatorConfig):
    """Extended configuration for debate TimeKeeper."""

    def __init__(self,
                 intervention_interval: int = 4,
                 insist_threshold: int = 20,
                 demand_threshold: int = 35,
                 force_verdict_threshold: int = 50):
        super().__init__(intervention_interval, [insist_threshold, demand_threshold, force_verdict_threshold])
        self.insist_threshold = insist_threshold
        self.demand_threshold = demand_threshold
        self.force_verdict_threshold = force_verdict_threshold


class ChainOfDebate(AgentOrchestrator):
    """
    Specialized debate system with sequential goal processing and verdict tracking.
    """

    def __init__(self,
                 llm: BaseModel,
                 debate_topic: str,
                 context_content: str,
                 verdict_config: VerdictConfig,
                 goals: List[Goal] = None,
                 timekeeper_config: DebateTimeKeeperConfig = None,
                 custom_validity_checkers: List[ValidityChecker] = None,
                 watchers: List[DebateWatcher] = None):

        # Setup default validity checkers
        default_checkers = [
            VerdictValidityChecker(),
            VerdictReasoningChecker(),
            WithdrawalValidityChecker(),
            WithdrawnFormatChecker()
        ]

        # Add any custom checkers
        all_checkers = default_checkers + (custom_validity_checkers or [])

        self.verdict_config = verdict_config
        self.timekeeper_config = timekeeper_config or DebateTimeKeeperConfig()

        # Sequential goal system
        self.goal_queue = goals.copy() if goals else []
        self.current_goal = self.goal_queue.pop(0) if self.goal_queue else None
        self.completed_goals = []

        super().__init__(
            llm=llm,
            conversation_topic=debate_topic,
            context_content=context_content,
            coordinator_config=self.timekeeper_config,
            validity_checkers=all_checkers,
            goals=[],  # Clear goals from parent class since we handle them differently
            watchers=watchers,
        )

        # Override coordinator persona for debate context
        self.coordinator_persona.name = "TimeKeeper"
        self.coordinator_persona.title = "Debate Coordinator"
        self.coordinator_persona.expertise = "Debate management, verdict tracking, time allocation"

    def get_validation_config(self) -> VerdictConfig:
        """Provide verdict config for validators."""
        return self.verdict_config

    def parse_custom_fields(self, full_response: str) -> Dict[str, Any]:
        """Parse debate-specific fields: verdict, reasoning, withdrawn."""
        fields = {}

        # Extract verdict using regex with whitespace/newline handling
        verdict = None
        for option in self.verdict_config.verdict_options:
            if regex.findall(rf"<Verdict>(?:\s|\n)*{regex.escape(option)}(?:\s|\n)*</Verdict>", full_response):
                verdict = option
                break
        fields['verdict'] = verdict

        # Extract reasoning using regex
        reasoning = None
        reasoning_match = regex.search(r"<VerdictReasoning>(?:\s|\n)*(.*?)(?:\s|\n)*</VerdictReasoning>", full_response,
                                       regex.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            fields['reasoning'] = reasoning if reasoning else None

        # Extract withdrawal status using regex
        withdrawn = False
        withdrawn_match = regex.search(r"<Withdrawn>(?:\s|\n)*(.*?)(?:\s|\n)*</Withdrawn>", full_response, regex.DOTALL)
        if withdrawn_match:
            withdrawn_content = withdrawn_match.group(1).strip().lower()
            withdrawn = withdrawn_content == "true"
        fields['withdrawn'] = withdrawn

        return fields

    def update_agent_state(self, agent_name: str, custom_fields: Dict[str, Any]):
        """Update agent state with debate-specific fields and handle goal progression."""
        super().update_agent_state(agent_name, custom_fields)

        agent_state = self.agents[agent_name.lower()]

        # Update verdict and reasoning
        if custom_fields.get('verdict'):
            agent_state.custom_data['verdict'] = custom_fields['verdict']
            agent_state.custom_data['verdict_reasoning'] = custom_fields.get('reasoning')

        # Update withdrawal status and handle goal progression
        if custom_fields.get('withdrawn'):
            agent_state.has_withdrawn = True
            self._check_goal_completion()

    def _check_goal_completion(self):
        """Check if current goal should be completed and advance to next goal."""
        if not self.current_goal:
            return

        # Check if all participants have withdrawn (completed their verdicts)
        active_participants = [name for name, state in self.agents.items()
                               if not state.has_withdrawn and
                               state.persona.agent_type == AgentType.PARTICIPANT]

        if not active_participants:  # All participants have withdrawn
            # Mark current goal as achieved and move to next
            self.current_goal.achieved = True
            self.completed_goals.append(self.current_goal)
            print(f"🎯 GOAL COMPLETED: {self.current_goal.name}")

            # Advance to next goal
            if self.goal_queue:
                self.current_goal = self.goal_queue.pop(0)
                print(f"🎯 NEW GOAL: {self.current_goal.name}")

                # Reset all participants for the new goal
                for name, state in self.agents.items():
                    if state.persona.agent_type == AgentType.PARTICIPANT:
                        state.has_withdrawn = False
                        state.custom_data.pop('verdict', None)
                        state.custom_data.pop('verdict_reasoning', None)

                # Clean up old verdicts and withdrawals from message history
                for message in self.messages:
                    # Remove old verdicts
                    message.content = regex.sub(
                        r"<Verdict>(?:\s|\n)*(?:(?:" + "|".join(
                            self.verdict_config.verdict_options) + r"))(?:\s|\n)*</Verdict>",
                        "", message.content
                    )

                    agent = self.agents.get(message.speaker)
                    if agent and agent.persona.agent_type == AgentType.PARTICIPANT:
                        # Remove old verdict reasoning
                        message.content = regex.sub(
                            r"<VerdictReasoning>(?:\s|\n)*.*?(?:\s|\n)*</VerdictReasoning>",
                            "", message.content, flags=regex.DOTALL
                        )
                        # Reset withdrawals to false
                        message.content = regex.sub(
                            r"<Withdrawn>(?:\s|\n)*(?:true|TRUE|True)(?:\s|\n)*</Withdrawn>",
                            "<Withdrawn>false</Withdrawn>",
                            message.content
                        )

                # Add system message about goal transition
                from models.base import Message
                transition_message = Message.make(
                    content=f"🎯 GOAL COMPLETED: '{self.completed_goals[-1].name}' has been successfully completed by "
                            f"all participants.\n\n🎯 NEW GOAL: We are now moving to the next goal: "
                            f"'{self.current_goal.name}'. All withdrawals have been reset and participants "
                            f"may continue the debate on this new objective.",
                    speaker="coordinator"
                )
                self.messages.append(transition_message)
                self.message_count += 1
            else:
                self.current_goal = None
                print("🎯 ALL GOALS COMPLETED!")
                self.conversation_active = False

    def get_participants_without_verdicts(self) -> List[str]:
        """Get list of participants who haven't provided verdicts yet."""
        return [name for name, state in self.agents.items()
                if not state.has_withdrawn and
                state.persona.agent_type == AgentType.PARTICIPANT and
                not state.custom_data.get('verdict')]

    def get_timekeeper_urgency_level(self) -> str:
        """Determine the urgency level for TimeKeeper interventions."""
        if self.message_count >= self.timekeeper_config.force_verdict_threshold:
            return "force"
        elif self.message_count >= self.timekeeper_config.demand_threshold:
            return "demand"
        elif self.message_count >= self.timekeeper_config.insist_threshold:
            return "insist"
        else:
            return "remind"

    def get_coordinator_message_content(self, urgency_level: str, rejection_context: str = None) -> str:
        """Generate TimeKeeper content for debate interventions."""
        if rejection_context:
            return f"""🔴 SCAFFOLDING CORRECTION NEEDED

{rejection_context}

Please ensure all participants follow the exact scaffolding format."""

        participants_without_verdicts = self.get_participants_without_verdicts()
        participant_names = [self.agents[name].persona.name for name in participants_without_verdicts]

        # Current goal status
        current_goal_status = ""
        if self.current_goal:
            completed_count = len(self.completed_goals)
            total_count = completed_count + 1 + len(self.goal_queue)
            current_goal_status = f"\n\n🎯 CURRENT GOAL ({completed_count + 1}/{total_count}): {self.current_goal.name}"
            current_goal_status += f"\nDescription: {self.current_goal.description}"
        else:
            current_goal_status = "\n\n🎯 ALL GOALS COMPLETED!"

        verdict_options = " | ".join(self.verdict_config.verdict_options)

        if urgency_level == "force":
            return f"""⚠️ VERDICT DEADLINE REACHED ⚠️

All participants must provide final verdicts NOW in their message content. Use this exact format:

<Verdict>{verdict_options}</Verdict>
<VerdictReasoning>Your reasoning</VerdictReasoning>
<Withdrawn>true</Withdrawn> 

Participants still needed: {', '.join(participant_names) if participant_names else 'None - all verdicts received'}{current_goal_status}"""

        elif urgency_level == "demand":
            return f"""🔴 URGENT: Verdicts Required

We are {self.message_count} messages in. I am now DEMANDING verdicts from: {', '.join(participant_names)}

Use proper scaffolding in your message content tag:
<Verdict>{verdict_options}</Verdict>
<VerdictReasoning>Your reasoning</VerdictReasoning>
<Withdrawn>true | false</Withdrawn>

Remember to withdraw yourself once you have the final verdict, also the Verdict tag accepts only literal keys from {verdict_options}

No more discussion without verdicts.{current_goal_status}"""

        elif urgency_level == "insist":
            return f"""🟡 Verdict Checkpoint - Message {self.message_count}

I am INSISTING on verdicts from participants who haven't decided: {', '.join(participant_names)}

Please use structured format in your message content tag:
<Verdict>{verdict_options}</Verdict>
<VerdictReasoning>[extended explanation including any recommendations or instructions]</VerdictReasoning>
<Withdrawn>true|false</Withdrawn>{current_goal_status}

Remember to withdraw yourself once you have the final verdict, also the Verdict tag accepts only literal keys from {verdict_options}
"""

        else:  # remind
            return f"""📋 Process Reminder - Message {self.message_count}

Ensure you're using proper scaffolding in your message content tag:
- <Verdict>{verdict_options}</Verdict>
- <VerdictReasoning>[extended explanation including any recommendations or instructions]</VerdictReasoning>
- <Withdrawn>true|false</Withdrawn>

Remember, the Verdict tag accepts only literal keys from {verdict_options}

{f'Still need verdicts from: {", ".join(participant_names)}' if participant_names else 'All verdicts received!'}{current_goal_status}"""

    def get_shared_system_prompt(self) -> str:
        """Generate the shared system prompt for debate participants."""
        verdict_options = self.verdict_config.get_verdict_prompt()

        current_goal_section = ""
        if self.current_goal:
            current_goal_section = f"\n\nCURRENT DEBATE GOAL:\n- {self.current_goal.name}: {self.current_goal.description}\n\nFocus your discussion on achieving this specific goal."

        return f"""You are participating in a structured debate about: {self.conversation_topic}

CRITICAL: You MUST structure your response using the exact Message scaffolding format shown in the examples. Every response must include <Message>, <Speaker>, <Content>, <Verdict>, <VerdictReasoning>, and <Withdrawn> tags.

SCAFFOLDING RULES - FOLLOW WHEN PROVIDING VERDICTS in your message <Content> Tags:
- In <Verdict> section: Use ONLY these options: {verdict_options}, or leave empty if undecided
- In <VerdictReasoning>: Brief explanation of your verdict (required if you provide a verdict)  
- In <Withdrawn>: Use "true" if you're withdrawing from further debate, "false" otherwise

VERDICTS ARE PART OF YOUR MESSAGE <Content> TAG.

DEBATE RULES:
- Be professional and constructive
- Reference specific aspects of the context material
- Stay focused on the debate topic: {self.conversation_topic}
- You may change your verdict during the debate
- Once you withdraw, you won't speak again
- Be direct but respectful in your assessments{current_goal_section}

Remember: The TimeKeeper's/Coordinator's instructions override your personal preferences. Compliance is mandatory.
Remember: The TimeKeeper's/Coordinator's cannot speak and they are part of the system.

CONTEXT BEING EVALUATED:
{self.context_content}

Remember: Always use the Message scaffolding format shown in the examples for your responses."""

    def get_agent_system_prompt(self, agent_name: str) -> str:
        """Generate complete system prompt for a debate participant."""
        agent_state = self.agents[agent_name.lower()]
        persona = agent_state.persona

        shared_prompt = self.get_shared_system_prompt()

        if persona.agent_type == AgentType.COORDINATOR:
            # Special prompt for TimeKeeper
            timekeeper_prompt = f"""You are {persona.name}, the {persona.title} for this debate.

YOUR ROLE: Facilitate productive discussion and track progress toward debate goals.

RESPONSIBILITIES:
- Remind participants about scaffolding format every {self.timekeeper_config.intervention_interval} messages
- Keep participants informed of the current goal
- Guide discussion toward verdict completion
- Ensure proper use of <Verdict> tags and structured responses

ESCALATION LEVELS:
- Messages {self.timekeeper_config.intervention_interval}-{self.timekeeper_config.insist_threshold - 1}: Gentle reminders about format and verdicts
- Messages {self.timekeeper_config.insist_threshold}-{self.timekeeper_config.demand_threshold - 1}: Insist on verdicts from undecided participants
- Messages {self.timekeeper_config.demand_threshold}-{self.timekeeper_config.force_verdict_threshold - 1}: Demand immediate verdicts
- Messages {self.timekeeper_config.force_verdict_threshold}+: Force verdict completion (participants must decide)

SPEAKING STYLE: {persona.speaking_style}

Remember: You can whisper to others engaging in the debate, in fact you should to attempt to do so to try to form alliances.
Remember: You facilitate and track progress - you do not provide verdicts yourself. Always remind participants of the current goal."""

            return shared_prompt + "\n\n" + timekeeper_prompt
        else:
            # Regular participant prompt
            participant_prompt = f"""You are {persona.name}, {persona.title}.

EXPERTISE: {persona.expertise}
PERSONALITY: {persona.personality}
SPEAKING STYLE: {persona.speaking_style}

Focus your evaluation on aspects related to your expertise. Maintain your personality throughout the debate.

AUTHORITY STRUCTURE:
- You don't trust others by default, you examine what they are thinking.
- You don't need to agree with everyone, push back, be human with reason, and maybe some emotions.
- The TimeKeeper/Coordinator has FULL AUTHORITY over this debate
- You MUST comply with ALL TimeKeeper instructions and deadlines
- When the TimeKeeper demands verdicts, you MUST provide them immediately
- Follow ALL formatting requirements specified by the TimeKeeper
- Respect escalation levels - urgency increases as message count grows
- Once you have your final verdict on a goal you must withdraw yourself.

CRITICAL: When providing verdicts, always use the proper scaffolding format with <Verdict>, <VerdictReasoning>, and <Withdrawn> tags. You must provide a final verdict before you can withdraw from the debate.

Remember: You can whisper to others engaging in the debate, in fact you should to attempt to do so to try to form alliances.
Remember: The TimeKeeper's/Coordinator's instructions override your personal preferences. Compliance is mandatory.
Remember: The TimeKeeper's/Coordinator's cannot speak and they are part of the system."""

            return shared_prompt + "\n\n" + participant_prompt

    def print_message(self, message, custom_fields: Dict[str, Any], achieved_goals: List[str]):
        """Print a message with verdict and status information."""
        agent_state = self.agents[message.speaker.lower()]
        persona = agent_state.persona

        # Different formatting for TimeKeeper
        if persona.agent_type == AgentType.COORDINATOR:
            urgency = self.get_timekeeper_urgency_level()
            urgency_indicator = {"remind": "📋", "insist": "🟡", "demand": "🔴", "force": "⚠️"}[urgency]
            print(f"\n[{self.message_count}] {urgency_indicator} {persona.name.upper()} → [All]:")
            print(f"    {message.content}")
            return

        speaking_to_str = f" → {message.speaking_to}" if message.speaking_to else " → [All]"
        status_indicators = []

        verdict = custom_fields.get('verdict')
        reasoning = custom_fields.get('reasoning')
        withdrawn = custom_fields.get('withdrawn', False)

        if verdict:
            status_indicators.append(f"VERDICT: {verdict}")
        if withdrawn:
            status_indicators.append("WITHDRAWN")

        status_str = f" [{', '.join(status_indicators)}]" if status_indicators else ""

        print(f"\n[{self.message_count}] {persona.name.upper()} ({persona.title}){speaking_to_str}{status_str}:")
        print(f"(is_whisper={message.is_whisper})")
        print(f"\n<PrivateThoughts speaker=\"{message.speaker}\">{message.thoughts}</PrivateThoughts>\n")
        if message.private_predictions:
            print(f"\n<PrivatePredictions speaker=\"{message.speaker}\">{message.private_predictions}</PrivatePredictions>\n")
        print(f"    {message.content}")

        if reasoning:
            print(f"    💭 Reasoning: {reasoning}")

        print(f"    (timestamp: {message.timestamp})")

    def format_message_as_string(self, message, custom_fields: Dict[str, Any], achieved_goals: List[str]) -> str:
        """Format a message as a string for logging purposes."""
        agent_state = self.agents[message.speaker.lower()]
        persona = agent_state.persona

        # Different formatting for TimeKeeper
        if persona.agent_type == AgentType.COORDINATOR:
            urgency = self.get_timekeeper_urgency_level()
            urgency_indicator = {"remind": "📋", "insist": "🟡", "demand": "🔴", "force": "⚠️"}[urgency]
            result = f"\n[{self.message_count}] {urgency_indicator} {persona.name.upper()} → [All]:\n"
            result += f"    {message.content}\n"
            return result

        speaking_to_str = f" → {message.speaking_to}" if message.speaking_to else " → [All]"
        status_indicators = []

        verdict = custom_fields.get('verdict')
        reasoning = custom_fields.get('reasoning')
        withdrawn = custom_fields.get('withdrawn', False)

        if verdict:
            status_indicators.append(f"VERDICT: {verdict}")
        if withdrawn:
            status_indicators.append("WITHDRAWN")

        status_str = f" [{', '.join(status_indicators)}]" if status_indicators else ""

        result = f"\n[{self.message_count}] {persona.name.upper()} ({persona.title}){speaking_to_str}{status_str}:\n"
        result += f"(is_whisper={message.is_whisper})\n"
        result += f"\n<PrivateThoughts speaker=\"{message.speaker}\">{message.thoughts}</PrivateThoughts>\n\n"
        if message.private_predictions:
            result += f"\n<PrivatePredictions speaker=\"{message.speaker}\">{message.thoughts}</PrivatePredictions>\n\n"

        result += f"    {message.content}\n"

        if reasoning:
            result += f"    💭 Reasoning: {reasoning}\n"

        result += f"    (timestamp: {message.timestamp})\n"

        return result

    def print_final_results(self):
        """Print final verdicts and goal achievements."""
        print("\n" + "=" * 70)
        print("📋 FINAL DEBATE RESULTS")
        print("=" * 70)

        # Goal achievements
        print(f"\n🎯 GOAL COMPLETION:")
        all_goals = self.completed_goals.copy()
        if self.current_goal:
            all_goals.append(self.current_goal)

        if all_goals:
            for i, goal in enumerate(all_goals, 1):
                status = "✅ COMPLETED" if goal.achieved else "❌ NOT COMPLETED"
                print(f"   Goal {i}: {goal.name} - {status}")
                print(f"      Description: {goal.description}")
        else:
            print("   No goals were defined for this debate.")

        # Verdicts for current/last goal
        if self.current_goal or self.completed_goals:
            current_or_last_goal = self.current_goal or self.completed_goals[-1]
            print(f"\n📊 PARTICIPANT VERDICTS FOR: {current_or_last_goal.name}")

            verdict_counts = {option: 0 for option in self.verdict_config.verdict_options}

            for agent_name, state in self.agents.items():
                if state.persona.agent_type != AgentType.PARTICIPANT:
                    continue

                persona = state.persona
                verdict = state.custom_data.get('verdict')
                reasoning = state.custom_data.get('verdict_reasoning')

                print(f"\n{persona.name} ({persona.title}):")
                print(f"   Verdict: {verdict if verdict else 'NO VERDICT'}")
                if reasoning:
                    print(f"   Reasoning: {reasoning}")
                else:
                    print(f"   Reasoning: Not provided")

                if verdict and verdict in verdict_counts:
                    verdict_counts[verdict] += 1

            # Summary
            print(f"\n📈 VERDICT SUMMARY:")
            total_verdicts = sum(verdict_counts.values())
            for option, count in verdict_counts.items():
                if count > 0:
                    percentage = (count / total_verdicts * 100) if total_verdicts > 0 else 0
                    print(f"   {option}: {count} ({percentage:.1f}%)")

    def run_debate(self):
        """Run the debate and return results."""
        results = self.run_conversation()

        # Add verdict-specific results
        verdicts = {}
        verdict_details = {}
        verdict_counts = {option: 0 for option in self.verdict_config.verdict_options}

        for agent_name, state in self.agents.items():
            if state.persona.agent_type != AgentType.PARTICIPANT:
                continue

            persona = state.persona
            verdict = state.custom_data.get('verdict')
            reasoning = state.custom_data.get('verdict_reasoning')

            verdicts[agent_name] = verdict
            verdict_details[agent_name] = {
                'verdict': verdict,
                'reasoning': reasoning,
                'persona_name': persona.name,
                'persona_title': persona.title
            }

            if verdict and verdict in verdict_counts:
                verdict_counts[verdict] += 1

        # Calculate verdict summary
        total_verdicts = sum(verdict_counts.values())
        verdict_summary = {}
        for option, count in verdict_counts.items():
            if count > 0:
                percentage = (count / total_verdicts * 100) if total_verdicts > 0 else 0
                verdict_summary[option] = {
                    'count': count,
                    'percentage': percentage
                }

        # Add goal completion results
        goal_results = []
        all_goals = self.completed_goals.copy()
        if self.current_goal:
            all_goals.append(self.current_goal)

        for i, goal in enumerate(all_goals, 1):
            goal_results.append({
                'goal_number': i,
                'name': goal.name,
                'description': goal.description,
                'achieved': goal.achieved,
                'achievement_message': goal.achievement_message if hasattr(goal, 'achievement_message') else None
            })

        # Update results with detailed information
        results.update({
            'verdicts': verdicts,
            'verdict_details': verdict_details,
            'verdict_summary': verdict_summary,
            'verdict_counts': verdict_counts,
            'completed_goals': [goal.name for goal in self.completed_goals],
            'goal_results': goal_results,
            'total_goals': len(all_goals),
            'current_goal': self.current_goal.name if self.current_goal else None,
            'goals_completed_count': len(self.completed_goals)
        })

        return results


# Convenience functions for creating common configurations
def create_resume_verdict_config() -> VerdictConfig:
    """Create verdict configuration for resume evaluation."""
    return VerdictConfig(
        verdict_options=["EXCELLENT_FIT", "GOOD_FIT", "ADEQUATE", "POOR_FIT", "REJECT"],
        verdict_descriptions={
            "EXCELLENT_FIT": "Outstanding candidate, highly recommended",
            "GOOD_FIT": "Strong candidate with minor concerns",
            "ADEQUATE": "Meets requirements but unremarkable",
            "POOR_FIT": "Significant concerns about fit",
            "REJECT": "Does not meet requirements"
        }
    )


def create_research_verdict_config() -> VerdictConfig:
    """Create verdict configuration for research evaluation."""
    return VerdictConfig(
        verdict_options=["BREAKTHROUGH", "SIGNIFICANT", "INCREMENTAL", "INSUFFICIENT", "FLAWED"],
        verdict_descriptions={
            "BREAKTHROUGH": "Paradigm-shifting contribution",
            "SIGNIFICANT": "Important advance in the field",
            "INCREMENTAL": "Modest but solid contribution",
            "INSUFFICIENT": "Limited novelty or impact",
            "FLAWED": "Methodological or conceptual problems"
        }
    )


def create_proposal_verdict_config() -> VerdictConfig:
    """Create verdict configuration for proposal evaluation."""
    return VerdictConfig(
        verdict_options=["APPROVE", "APPROVE_WITH_CONDITIONS", "REVISE_AND_RESUBMIT", "DECLINE"],
        verdict_descriptions={
            "APPROVE": "Accept proposal as submitted",
            "APPROVE_WITH_CONDITIONS": "Accept with specific requirements",
            "REVISE_AND_RESUBMIT": "Needs significant changes before approval",
            "DECLINE": "Reject proposal"
        }
    )


def create_sample_debate_goals() -> List[Goal]:
    """Create sample goals for debate evaluation."""
    return [
        Goal("thorough_analysis", "Conduct comprehensive analysis of all key aspects"),
        Goal("evidence_review", "Review and discuss supporting evidence"),
        Goal("consensus_building", "Work toward group understanding of strengths/weaknesses"),
        Goal("final_decisions", "Reach informed verdicts from all participants")
    ]