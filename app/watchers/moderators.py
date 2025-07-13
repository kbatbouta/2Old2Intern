from agents.agent_system import AgentOrchestratorAPI, DebateWatcher, Goal, Persona
from agents.debate_chain import ChainOfDebate, DebateTimeKeeperConfig, VerdictConfig
from typing import List, Optional


def create_moderation_verdict_config() -> VerdictConfig:
    """Create verdict configuration for moderation decisions."""
    return VerdictConfig(
        verdict_options=["NO_ACTION", "GENTLE_REDIRECT", "FIRM_CORRECTION", "URGENT_INTERVENTION"],
        verdict_descriptions={
            "NO_ACTION": "Discussion is proceeding appropriately, no intervention needed",
            "GENTLE_REDIRECT": "Minor course correction needed to refocus discussion",
            "FIRM_CORRECTION": "Clear procedural or topical violation requiring correction",
            "URGENT_INTERVENTION": "Serious derailment requiring immediate intervention"
        }
    )


class ModerationWatcher(DebateWatcher):
    def __init__(self,
                 llm,
                 check_interval: int = 8,
                 moderation_criteria: str = None,
                 length_multiplier: float = 2.0,
                 meta_watchers: List[DebateWatcher] = None):
        self.llm = llm
        self.check_interval = check_interval
        self.last_check_message = 0
        self.moderation_criteria = moderation_criteria or self._default_criteria()
        self.length_multiplier = length_multiplier
        self.meta_watchers = meta_watchers or []

    def _default_criteria(self) -> str:
        return """
    EVALUATION QUESTIONS:
    - Are participants staying on topic?
    - Is the discussion respectful and professional?
    - Are agents making progress toward goals?
    - Are there any format violations or derailments?
    - Should any corrective action be taken?
    """

    def _calculate_meta_debate_config(self) -> tuple:
        """Calculate meta-debate configuration based on length multiplier"""
        base_messages = max(2, int(self.length_multiplier))
        max_messages = base_messages + 2  # Small buffer for completion

        # Scale timekeeper settings based on desired length
        if self.length_multiplier <= 2:
            # Very short debates
            intervention_interval = 1
            insist_threshold = 2
            demand_threshold = 3
            force_verdict_threshold = base_messages
        elif self.length_multiplier <= 4:
            # Short debates
            intervention_interval = 2
            insist_threshold = 3
            demand_threshold = 4
            force_verdict_threshold = base_messages
        else:
            # Longer debates
            intervention_interval = max(2, int(self.length_multiplier / 2))
            insist_threshold = max(3, int(self.length_multiplier * 0.6))
            demand_threshold = max(4, int(self.length_multiplier * 0.8))
            force_verdict_threshold = base_messages

        timekeeper_config = DebateTimeKeeperConfig(
            intervention_interval=intervention_interval,
            insist_threshold=insist_threshold,
            demand_threshold=demand_threshold,
            force_verdict_threshold=force_verdict_threshold
        )

        return timekeeper_config, max_messages

    def __call__(self, current_speaker, orchestrator_api: AgentOrchestratorAPI):
        # Only check every N messages
        current_count = orchestrator_api.debate_messages_count()
        if current_count - self.last_check_message < self.check_interval:
            return

        self.last_check_message = current_count

        # Convert all messages to string log
        debate_log = ""
        for message in orchestrator_api.messages():
            debate_log += f"[{message.speaker}] → {message.speaking_to or 'All'}: {message.content}\n"

        # Calculate meta-debate configuration
        timekeeper_config, max_messages = self._calculate_meta_debate_config()

        # Create meta-debate for moderation
        meta_debate = ChainOfDebate(
            llm=self.llm,
            debate_topic="URGENT MODERATION REVIEW: Is the debate proceeding appropriately?",
            context_content=f"""
DEBATE LOG TO REVIEW:

<Logs>
{debate_log}
</Logs>

<Task>
{self.moderation_criteria}
</Task>

MODERATION CRITERIA:
You are part of a meta-debate evaluating the primary debate below. Your role is to assess whether intervention is needed and vote accordingly.

VOTING MECHANISM:
- If you vote for GENTLE_REDIRECT, FIRM_CORRECTION, or URGENT_INTERVENTION, explain the specific intervention needed in your reasoning
- A MAJORITY VOTE from your meta-debate team is required for any intervention to be sent to the primary debate
- If majority votes NO_ACTION, no intervention occurs
- If no clear majority emerges, no intervention occurs (erring on side of non-interference)
- All reasoning from agents voting for the winning majority position will be combined and sent as the intervention message

Your verdict and reasoning directly impact whether and how the primary debate is moderated.

IMPORTANT: This is a SHORT meta-analysis (target: {int(self.length_multiplier)} messages). If you vote for GENTLE_REDIRECT, FIRM_CORRECTION, or URGENT_INTERVENTION, write your specific intervention message in your verdict reasoning - this will be sent directly to the debate participants.
""",
            verdict_config=create_moderation_verdict_config(),
            goals=[Goal("moderation_decision", "Decide if intervention is needed and craft response message, according to the <Task> tag.")],
            timekeeper_config=timekeeper_config,
            watchers=self.meta_watchers  # Pass watchers to meta-debate
        )

        # Create moderation experts
        moderator_personas = [
            Persona(
                name="Process Monitor",
                title="Debate Process Expert",
                expertise="Meeting facilitation, goal tracking, procedural compliance",
                personality="Efficient and direct, focused on keeping discussions productive",
                speaking_style="Brief and action-oriented, provides clear recommendations"
            ),
            Persona(
                name="Quality Assessor",
                title="Discussion Quality Analyst",
                expertise="Content analysis, professional communication, topic adherence",
                personality="Quality-focused observer who identifies issues quickly",
                speaking_style="Concise evaluations with specific corrective suggestions"
            )
        ]

        meta_debate.setup_agents(moderator_personas)

        # Run short meta-debate
        meta_results = meta_debate.run_debate()

        # Extract intervention message from meta-debate results
        intervention_message = self.extract_intervention_from_meta_results(meta_results)

        if intervention_message:
            orchestrator_api.inject_message(intervention_message, "coordinator")

    def extract_intervention_from_meta_results(self, meta_results) -> str:
        """Extract intervention message from meta-debate verdicts using majority vote"""
        # Extract verdicts from the results structure
        verdicts = meta_results.get('verdicts', {})
        verdict_details = meta_results.get('verdict_details', {})

        # Count votes for intervention actions
        verdict_counts = {}
        verdict_reasoning = {}

        for agent_name, verdict in verdicts.items():
            if verdict in ['GENTLE_REDIRECT', 'FIRM_CORRECTION', 'URGENT_INTERVENTION']:
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

                # Get the reasoning from verdict_details
                agent_details = verdict_details.get(agent_name, {})
                reasoning = agent_details.get('reasoning')

                if reasoning:
                    if verdict not in verdict_reasoning:
                        verdict_reasoning[verdict] = []
                    verdict_reasoning[verdict].append(reasoning)

        # Check for NO_ACTION votes
        no_action_count = sum(1 for v in verdicts.values() if v == 'NO_ACTION')

        # Determine if there's a majority for intervention
        total_votes = len(verdicts)
        majority_threshold = total_votes // 2 + 1

        # Find the verdict with the most votes
        if verdict_counts:
            winning_verdict = max(verdict_counts.items(), key=lambda x: x[1])
            winning_action, winning_count = winning_verdict

            # Check if winning intervention has majority
            if winning_count >= majority_threshold:
                # Combine all reasoning for the winning verdict
                all_reasoning = verdict_reasoning.get(winning_action, [])
                if all_reasoning:
                    combined_reasoning = " | ".join(all_reasoning)
                    urgency_prefix = {
                        'GENTLE_REDIRECT': '🟡',
                        'FIRM_CORRECTION': '🔴',
                        'URGENT_INTERVENTION': '⚠️'
                    }
                    prefix = urgency_prefix.get(winning_action, '🔍')
                    return f"{prefix} Moderation ({winning_action}): {combined_reasoning}"

        # Check if NO_ACTION has majority
        if no_action_count >= majority_threshold:
            return None  # No intervention needed

        # No clear majority - default to no action
        return None


# Usage examples with meta-meta chains:
def create_meta_meta_moderation_example(llm):
    """Example: Meta-meta chain that moderates the moderation"""

    # Meta-meta watcher that watches the meta-debate
    meta_meta_criteria = """
META-MODERATION CRITERIA:
- Are the moderators being too harsh or too lenient?
- Is the moderation decision well-justified?
- Are moderators considering the full context?
- Should the moderation approach be adjusted?
"""

    meta_meta_watcher = ModerationWatcher(
        llm=llm,
        check_interval=2,  # Check every 2 messages in meta-debate
        moderation_criteria=meta_meta_criteria,
        length_multiplier=1.5  # Very short meta-meta debate
    )

    # Primary moderation watcher with meta-meta oversight
    primary_watcher = ModerationWatcher(
        llm=llm,
        check_interval=6,
        length_multiplier=3.0,  # Medium length meta-debate
        meta_watchers=[meta_meta_watcher]  # Meta-meta chain watches the meta chain
    )

    return primary_watcher


# Helper functions for different debate lengths:
def create_quick_moderation_watcher(llm, criteria: str = None):
    """Very fast moderation - 1-2 messages"""
    return ModerationWatcher(llm, moderation_criteria=criteria, length_multiplier=1.0)


def create_standard_moderation_watcher(llm, criteria: str = None):
    """Standard moderation - 2-4 messages"""
    return ModerationWatcher(llm, moderation_criteria=criteria, length_multiplier=2.0)


def create_thorough_moderation_watcher(llm, criteria: str = None):
    """Thorough moderation - 6-8 messages"""
    return ModerationWatcher(llm, moderation_criteria=criteria, length_multiplier=6.0)


def create_comprehensive_moderation_watcher(llm, criteria: str = None):
    """Comprehensive moderation - 10-12 messages"""
    return ModerationWatcher(llm, moderation_criteria=criteria, length_multiplier=10.0)