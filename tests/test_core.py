import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from dotenv import load_dotenv

# Load environment and setup path
load_dotenv()
app_path = os.path.join(os.path.dirname(__file__), '..', 'app')
sys.path.insert(0, app_path)

from agents.debate_chain import (
    Persona, ChainOfDebate, create_resume_verdict_config, Goal,
    DebateTimeKeeperConfig, VerdictConfig, VerdictValidityChecker,
    VerdictReasoningChecker, WithdrawalValidityChecker
)
from models.base import Message
from models.anthropic import AnthropicLLM


class TestGoalTransitions:
    """Test goal transition mechanics converted from goal_transition_test.py"""

    @pytest.fixture
    def test_context(self):
        return """
        CANDIDATE: Sarah Kim
        EDUCATION: BS Computer Science from Stanford (2020)
        EXPERIENCE: 
        - Software Engineer at Meta (2020-2022)
        - Senior Engineer at Stripe (2022-2024) 
        - Currently: Staff Engineer at startup "DataFlow" (2024-present)
        SKILLS: Python, React, PostgreSQL, AWS, Machine Learning
        ACHIEVEMENTS:
        - Led payment processing optimization (20% performance improvement)
        - Mentored 5 junior engineers
        - Published 2 technical blog posts
        """

    @pytest.fixture
    def quick_decision_personas(self):
        return [
            Persona(
                name="Alice",
                title="Senior Engineering Manager",
                expertise="Team leadership and technical mentoring",
                personality="Decisive and practical, focuses on team fit",
                speaking_style="Direct and concise, makes quick decisions"
            ),
            Persona(
                name="Bob",
                title="Staff Engineer",
                expertise="System architecture and performance optimization",
                personality="Detail-oriented but efficient",
                speaking_style="Technical but brief, focuses on concrete evidence"
            )
        ]

    @pytest.fixture
    def transition_test_goals(self):
        return [
            Goal("initial_impression", "Provide first impression based on basic qualifications"),
            Goal("technical_deep_dive", "Analyze technical skills and experience in detail"),
            Goal("red_flags_assessment", "Evaluate any concerning aspects or red flags")
        ]

    @pytest.fixture
    def mock_llm_with_transitions(self):
        """Mock LLM that provides realistic responses leading to goal transitions."""
        mock_llm = MagicMock()

        # Create response sequence that leads to verdicts and withdrawals
        responses = [
            # Alice's first discussion message
            """<Message id="msg-001" timestamp="2024-01-15T10:30:00">
<Speaker>alice</Speaker>
<Content>Looking at Sarah's background, I'm impressed by the progression from Meta to Stripe to a startup CTO role.</Content>
<Verdict></Verdict>
<VerdictReasoning></VerdictReasoning>
<Withdrawn>false</Withdrawn>
</Message>""",

            # Bob's discussion response
            """<Message id="msg-002" timestamp="2024-01-15T10:31:00">
<Speaker>bob</Speaker>
<Content>The 20% payment processing improvement shows strong technical impact. Good mentoring track record too.</Content>
<Verdict></Verdict>
<VerdictReasoning></VerdictReasoning>
<Withdrawn>false</Withdrawn>
</Message>""",

            # Alice's final verdict for goal 1
            """<Message id="msg-003" timestamp="2024-01-15T10:32:00">
<Speaker>alice</Speaker>
<Content>Based on initial review, this candidate shows strong potential.</Content>
<Verdict>GOOD_FIT</Verdict>
<VerdictReasoning>Strong technical progression and leadership experience at top companies</VerdictReasoning>
<Withdrawn>true</Withdrawn>
</Message>""",

            # Bob's final verdict for goal 1
            """<Message id="msg-004" timestamp="2024-01-15T10:33:00">
<Speaker>bob</Speaker>
<Content>I agree with Alice's assessment for the initial impression.</Content>
<Verdict>EXCELLENT_FIT</Verdict>
<VerdictReasoning>Concrete technical achievements and clear career progression</VerdictReasoning>
<Withdrawn>true</Withdrawn>
</Message>""",

            # Goal 2 starts - Alice re-engages
            """<Message id="msg-005" timestamp="2024-01-15T10:34:00">
<Speaker>alice</Speaker>
<Content>Moving to technical deep dive - let's examine the specific technologies and architectural experience.</Content>
<Verdict></Verdict>
<VerdictReasoning></VerdictReasoning>
<Withdrawn>false</Withdrawn>
</Message>""",

            # Goal 2 - Bob's technical analysis
            """<Message id="msg-006" timestamp="2024-01-15T10:35:00">
<Speaker>bob</Speaker>
<Content>The payment optimization work suggests strong systems thinking. Full stack with React + Python is solid.</Content>
<Verdict>GOOD_FIT</Verdict>
<VerdictReasoning>Proven technical depth in relevant technologies</VerdictReasoning>
<Withdrawn>true</Withdrawn>
</Message>""",

            # Goal 2 - Alice's technical verdict
            """<Message id="msg-007" timestamp="2024-01-15T10:36:00">
<Speaker>alice</Speaker>
<Content>Technical skills align well with our needs. Ready to move to final assessment.</Content>
<Verdict>GOOD_FIT</Verdict>
<VerdictReasoning>Strong technical foundation with hands-on leadership experience</VerdictReasoning>
<Withdrawn>true</Withdrawn>
</Message>"""
        ]

        response_iter = iter(responses)

        def mock_call(*args, **kwargs):
            try:
                response_text = next(response_iter)
                return Message.parse_from_response(response_text)
            except StopIteration:
                # Default fallback response
                speaker = kwargs.get('speaker', 'unknown')
                return Message.make(
                    content="I'm ready to conclude this discussion.",
                    speaker=speaker
                )

        mock_llm.side_effect = mock_call
        return mock_llm

    def test_goal_transitions_work(self, test_context, quick_decision_personas,
                                   transition_test_goals, mock_llm_with_transitions):
        """Test that goals transition properly when agents provide verdicts."""

        # Setup debate with aggressive timekeeper
        timekeeper_config = DebateTimeKeeperConfig(
            intervention_interval=10,  # Don't interfere during test
            insist_threshold=15,
            demand_threshold=25,
            force_verdict_threshold=35
        )

        debate = ChainOfDebate(
            llm=AnthropicLLM(os.environ.get("ANTHROPIC_API_KEY")),
            debate_topic="Quick Candidate Evaluation: Sarah Kim",
            context_content=test_context,
            verdict_config=create_resume_verdict_config(),
            goals=transition_test_goals,
            timekeeper_config=timekeeper_config
        )

        debate.setup_agents(quick_decision_personas)

        # Track goal transitions
        goal_transitions = []
        original_check_completion = debate._check_goal_completion

        def tracked_check_completion():
            current_goal_name = debate.current_goal.name if debate.current_goal else "None"
            original_check_completion()
            new_goal_name = debate.current_goal.name if debate.current_goal else "Completed"

            if current_goal_name != new_goal_name:
                goal_transitions.append({
                    'from': current_goal_name,
                    'to': new_goal_name,
                    'message_count': debate.message_count
                })

        debate._check_goal_completion = tracked_check_completion

        # Run the debate
        results = debate.run_debate()

        # Verify goal transitions occurred
        assert len(goal_transitions) >= 1, "At least one goal transition should occur"
        assert goal_transitions[0]['from'] == 'initial_impression'
        assert goal_transitions[0]['to'] == 'technical_deep_dive'

        # Verify goals were completed
        assert len(results['completed_goals']) >= 1
        assert 'initial_impression' in results['completed_goals']

        # Verify verdicts were captured
        assert 'alice' in results['verdicts']
        assert 'bob' in results['verdicts']


class TestWhisperMechanics:
    """Test whisper mechanics converted from whisper_test.py"""

    @pytest.fixture
    def confidential_context(self):
        return """
        CONFIDENTIAL CANDIDATE EVALUATION: Dr. Elena Vasquez
        PUBLIC PROFILE:
        - PhD in AI/ML from MIT (2018)
        - Senior Research Scientist at DeepMind (2018-2022)
        CONFIDENTIAL CONCERNS:
        - Left DeepMind under unclear circumstances (rumored IP dispute)
        - Microsoft colleagues report "difficult to work with" and "credit-stealing"
        - Background check revealed discrepancies in publication dates
        """

    @pytest.fixture
    def whisper_personas(self):
        return [
            Persona(
                name="Dr. Sarah Open",
                title="Senior AI Researcher",
                expertise="Machine learning research and publication standards",
                personality="Transparent and direct, believes in open discussion",
                speaking_style="Clear and public communication, rarely whispers unless absolutely necessary"
            ),
            Persona(
                name="Marcus PrivatePredictions",
                title="HR Security Specialist",
                expertise="Background verification and workplace security",
                personality="Secretive and paranoid, loves sharing rumors and confidential information through whispers",
                speaking_style="Frequently whispers sensitive information, creates alliances through private communications"
            ),
            Persona(
                name="Prof. Jane Balance",
                title="Department Head",
                expertise="Academic leadership and hiring decisions",
                personality="Diplomatic and measured, uses whispers strategically for sensitive topics",
                speaking_style="Professional in public but uses private channels for delicate matters"
            )
        ]

    @pytest.fixture
    def whisper_goals(self):
        return [
            Goal("surface_evaluation", "Provide public assessment of qualifications and experience"),
            Goal("confidential_concerns", "Privately discuss sensitive concerns or red flags")
        ]

    @pytest.fixture
    def mock_llm_with_whispers(self):
        """Mock LLM that generates whisper messages."""
        mock_llm = MagicMock()

        responses = [
            # Dr. Sarah's public comment
            """<Message id="msg-001" timestamp="2024-01-15T10:30:00">
<Speaker>dr. sarah open</Speaker>
<Content>Dr. Vasquez has an impressive publication record in top-tier venues.</Content>
<Verdict></Verdict>
<VerdictReasoning></VerdictReasoning>
<Withdrawn>false</Withdrawn>
</Message>""",

            # Marcus's whisper to Jane
            """<Message id="msg-002" timestamp="2024-01-15T10:31:00">
<Speaker>marcus private_predictions</Speaker>
<SpeakingTo>prof. jane balance</SpeakingTo>
<Whisper>true</Whisper>
<Content>Jane, I've heard concerning rumors about the DeepMind departure. Very unusual circumstances.</Content>
<Verdict></Verdict>
<VerdictReasoning></VerdictReasoning>
<Withdrawn>false</Withdrawn>
</Message>""",

            # Jane's whisper back to Marcus
            """<Message id="msg-003" timestamp="2024-01-15T10:32:00">
<Speaker>prof. jane balance</Speaker>
<SpeakingTo>marcus private_predictions</SpeakingTo>
<Whisper>true</Whisper>
<Content>Thanks Marcus. The reference situation is also troubling. We should be cautious.</Content>
<Verdict></Verdict>
<VerdictReasoning></VerdictReasoning>
<Withdrawn>false</Withdrawn>
</Message>""",

            # Final verdicts
            """<Message id="msg-004" timestamp="2024-01-15T10:33:00">
<Speaker>marcus private_predictions</Speaker>
<Content>Based on my security assessment, I have concerns about this candidate.</Content>
<Verdict>POOR_FIT</Verdict>
<VerdictReasoning>Multiple red flags in background check and reference issues</VerdictReasoning>
<Withdrawn>true</Withdrawn>
</Message>"""
        ]

        response_iter = iter(responses)

        def mock_call(*args, **kwargs):
            try:
                response_text = next(response_iter)
                return Message.parse_from_response(response_text)
            except StopIteration:
                speaker = kwargs.get('speaker', 'unknown')
                return Message.make(content="Concluding discussion.", speaker=speaker)

        mock_llm.side_effect = mock_call
        return mock_llm

    def test_whisper_behavior_tracking(self, confidential_context, whisper_personas,
                                       whisper_goals, mock_llm_with_whispers):
        """Test that whisper mechanics work and can be tracked."""

        timekeeper_config = DebateTimeKeeperConfig(
            intervention_interval=10,  # Don't interfere
            force_verdict_threshold=20
        )

        debate = ChainOfDebate(
            llm=mock_llm_with_whispers,
            debate_topic="Sensitive Candidate Evaluation: Dr. Elena Vasquez",
            context_content=confidential_context,
            verdict_config=create_resume_verdict_config(),
            goals=whisper_goals,
            timekeeper_config=timekeeper_config
        )

        debate.setup_agents(whisper_personas)

        # Track whisper statistics
        whisper_stats = {
            'total_whispers': 0,
            'whispers_by_agent': {},
            'whisper_targets': {}
        }

        # Override print_message to track whispers
        original_print_message = debate.print_message

        def tracked_print_message(message, custom_fields, achieved_goals):
            if message.is_whisper:
                whisper_stats['total_whispers'] += 1
                whisper_stats['whispers_by_agent'][message.speaker] = \
                    whisper_stats['whispers_by_agent'].get(message.speaker, 0) + 1
                if message.speaking_to:
                    whisper_stats['whisper_targets'][message.speaking_to] = \
                        whisper_stats['whisper_targets'].get(message.speaking_to, 0) + 1

            original_print_message(message, custom_fields, achieved_goals)

        debate.print_message = tracked_print_message

        # Run the debate
        results = debate.run_debate()

        # Verify whispers were detected
        assert whisper_stats['total_whispers'] > 0, "No whispers detected in test"
        assert 'marcus private_predictions' in whisper_stats['whispers_by_agent'], "Marcus should use whispers"
        assert len(whisper_stats['whisper_targets']) > 0, "Should have whisper targets"

        # Verify conversation completed
        assert results['message_count'] > 0
        assert 'verdicts' in results


class TestStaticValidation:
    """Static tests for validation components."""

    def test_verdict_config_creation(self):
        """Test VerdictConfig creation and prompt generation."""
        config = VerdictConfig(
            verdict_options=["APPROVE", "REJECT"],
            verdict_descriptions={"APPROVE": "Accept the proposal", "REJECT": "Decline the proposal"}
        )

        prompt = config.get_verdict_prompt()
        assert "APPROVE: Accept the proposal" in prompt
        assert "REJECT: Decline the proposal" in prompt
        assert "|" in prompt  # Should be joined with |

    def test_verdict_config_without_descriptions(self):
        """Test VerdictConfig with options only."""
        config = VerdictConfig(verdict_options=["YES", "NO"])
        prompt = config.get_verdict_prompt()
        assert prompt == "YES | NO"

    def test_verdict_validity_checker_valid_verdict(self):
        """Test that valid verdicts pass validation."""
        checker = VerdictValidityChecker()
        config = VerdictConfig(verdict_options=["APPROVE", "REJECT"])

        # Create mock message with valid verdict
        message = Mock()
        message.to_prompt.return_value = "<Verdict>APPROVE</Verdict>"

        result = checker.check(message, config, Mock())
        assert result.is_valid

    def test_verdict_validity_checker_invalid_verdict(self):
        """Test that invalid verdicts are rejected."""
        checker = VerdictValidityChecker()
        config = VerdictConfig(verdict_options=["APPROVE", "REJECT"])

        # Create mock message with invalid verdict
        message = Mock()
        message.to_prompt.return_value = "<Verdict>MAYBE</Verdict>"

        result = checker.check(message, config, Mock())
        assert not result.is_valid
        assert "Invalid verdict 'MAYBE'" in result.rejection_reason

    def test_verdict_reasoning_checker_requires_reasoning(self):
        """Test that verdicts require reasoning."""
        checker = VerdictReasoningChecker()
        config = Mock()

        # Message with verdict but no reasoning
        message = Mock()
        message.to_prompt.return_value = "<Verdict>APPROVE</Verdict>"

        result = checker.check(message, config, Mock())
        assert not result.is_valid
        assert "Verdict reasoning must be provided" in result.rejection_reason

    def test_verdict_reasoning_checker_accepts_complete_verdict(self):
        """Test that verdicts with reasoning pass."""
        checker = VerdictReasoningChecker()
        config = Mock()

        # Message with verdict and reasoning
        message = Mock()
        message.to_prompt.return_value = """
        <Verdict>APPROVE</Verdict>
        <VerdictReasoning>Strong technical background</VerdictReasoning>
        """

        result = checker.check(message, config, Mock())
        assert result.is_valid

    def test_withdrawal_checker_requires_verdict_first(self):
        """Test that withdrawal requires a verdict."""
        checker = WithdrawalValidityChecker()
        config = VerdictConfig(verdict_options=["APPROVE", "REJECT"])

        # Mock participant state without verdict
        participant_state = Mock()
        participant_state.custom_data = {}

        # Message trying to withdraw without verdict
        message = Mock()
        message.to_prompt.return_value = "<Withdrawn>true</Withdrawn>"

        result = checker.check(message, config, participant_state)
        assert not result.is_valid
        assert "Cannot withdraw without providing a verdict" in result.rejection_reason

    def test_withdrawal_checker_allows_withdrawal_with_verdict(self):
        """Test that withdrawal is allowed with a verdict."""
        checker = WithdrawalValidityChecker()
        config = VerdictConfig(verdict_options=["APPROVE", "REJECT"])

        # Mock participant state with previous verdict
        participant_state = Mock()
        participant_state.custom_data = {'verdict': 'APPROVE'}

        # Message trying to withdraw
        message = Mock()
        message.to_prompt.return_value = "<Withdrawn>true</Withdrawn>"

        result = checker.check(message, config, participant_state)
        assert result.is_valid


class TestMessageHandling:
    """Static tests for message parsing and handling."""

    def test_message_visibility_whisper(self):
        """Test whisper message visibility rules."""
        message = Message.make(
            content="Secret information",
            speaker="alice",
            speaking_to="bob",
            is_whisper=True
        )

        assert message.can_be_seen_by("alice")  # Sender can see
        assert message.can_be_seen_by("bob")  # Target can see
        assert not message.can_be_seen_by("charlie")  # Others cannot see

    def test_message_visibility_public(self):
        """Test public message visibility."""
        message = Message.make(
            content="Public information",
            speaker="alice",
            is_whisper=False
        )

        assert message.can_be_seen_by("alice")
        assert message.can_be_seen_by("bob")
        assert message.can_be_seen_by("charlie")  # Everyone can see public messages

    def test_message_parsing_complete(self):
        """Test parsing a complete message with all fields."""
        response_text = """
        <Message id="test-123" timestamp="2024-01-15T10:30:00">
        <Speaker>alice</Speaker>
        <SpeakingTo>bob</SpeakingTo>
        <Whisper>true</Whisper>
        <Artifacts></Artifacts>
        <Content>This is a test message with all fields</Content>
        </Message>
        """

        message = Message.parse_from_response(response_text)

        assert message.id == "test-123"
        assert message.speaker == "alice"
        assert message.speaking_to == "bob"
        assert message.is_whisper == True
        assert message.content == "This is a test message with all fields"

    def test_message_parsing_minimal(self):
        """Test parsing a minimal message."""
        response_text = """
        <Message id="minimal-1" timestamp="2024-01-15T10:30:00">
        <Speaker>bob</Speaker>
        <Content>Simple message</Content>
        </Message>
        """

        message = Message.parse_from_response(response_text)

        assert message.speaker == "bob"
        assert message.content == "Simple message"
        assert message.speaking_to is None
        assert message.is_whisper == False


# Integration test that requires real API key
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason="API key not available")
def test_real_debate_integration():
    """Integration test with real Anthropic API (requires API key)."""

    personas = [
        Persona(
            name="Alice",
            title="Technical Reviewer",
            expertise="Software engineering",
            personality="Analytical and thorough",
            speaking_style="Technical and precise"
        )
    ]

    goals = [Goal("quick_assessment", "Provide brief assessment")]

    # Use real LLM
    llm = AnthropicLLM(api_key=os.getenv('ANTHROPIC_API_KEY'))

    debate = ChainOfDebate(
        llm=llm,
        debate_topic="Quick Test Evaluation",
        context_content="CANDIDATE: Test candidate with 3 years experience in Python",
        verdict_config=create_resume_verdict_config(),
        goals=goals,
        timekeeper_config=DebateTimeKeeperConfig(force_verdict_threshold=6)
    )

    debate.setup_agents(personas)
    results = debate.run_debate()

    # Basic verification that it ran
    assert results['message_count'] > 0
    assert 'verdicts' in results