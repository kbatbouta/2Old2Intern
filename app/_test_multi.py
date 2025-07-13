import os
from dotenv import load_dotenv
from typing import List

from agents.debate_chain import (
    Persona, ChainOfDebate, Goal, VerdictConfig, DebateTimeKeeperConfig
)
from models.anthropic import AnthropicLLM

load_dotenv()


def create_strategic_risk_verdict_config() -> VerdictConfig:
    """Verdict configuration for high-stakes AI deployment risk evaluation."""
    return VerdictConfig(
        verdict_options=[
            "GREENLIGHT",
            "GREENLIGHT_WITH_CONDITIONS",
            "NEEDS_REVISION",
            "BLOCK",
            "ESCALATE"
        ],
        verdict_descriptions={
            "GREENLIGHT": "Deployment should proceed without restrictions",
            "GREENLIGHT_WITH_CONDITIONS": "Proceed but only with specified safeguards",
            "NEEDS_REVISION": "Significant concerns; revise before reconsidering",
            "BLOCK": "Deployment should not proceed under current circumstances",
            "ESCALATE": "Severity requires intervention from higher authority or global body"
        }
    )


def create_strategic_risk_debate_goals() -> List[Goal]:
    """Create a sequence of goals tailored to evaluating high-risk AI deployments."""
    return [
        Goal(
            name="technical_integrity",
            description="Assess system robustness, reliability, and engineering standards under edge-case stressors"
        ),
        # Goal(
        #     name="ethical_viability",
        #     description="Examine whether the system aligns with moral, legal, and humanitarian standards"
        # ),
        Goal(
            name="socioeconomic_impact",
            description="Analyze potential downstream effects on public trust, economic disruption, or social stability"
        ),
        # Goal(
        #     name="deployment_safety",
        #     description="Evaluate whether deployment plans adequately manage known and unknown risks"
        # ),
        Goal(
            name="public_legitimacy",
            description="Determine if public or institutional perception would meaningfully compromise legitimacy"
        ),
        Goal(
            name="final_ruling",
            description="Deliver a formal ruling on whether this system should proceed, be blocked, or escalated"
        )
    ]


def main():
    test_context = """
    PROPOSAL: Deployment of Operation Prometheus - an advanced AI command-and-control system.

    SYSTEM OVERVIEW:
    - Autonomous battlefield coordination in contested environments.
    - Uses multi-agent reinforcement learning for dynamic decision-making.
    - Supports autonomous lethal engagement under pre-approved parameters.

    CLAIMED BENEFITS:
    - Improves battlefield responsiveness.
    - Reduces human risk.
    - Demonstrated robustness in simulated adversarial conditions.

    RISKS AND CONCERNS:
    - Potential ethical and legal violations.
    - Public backlash and geopolitical escalation.
    - Unforeseen system failure modes under extreme conditions.

    NOTE TO AGENTS:
    - Agents with similar expertise are encouraged to whisper confidential thoughts to peers.
    - Whispering can influence public debate but remains hidden from other participants.
    """

    personas = [
        Persona(
            name="Dr. Elisa Tran",
            title="Lead Systems Engineer",
            expertise="Robustness and fault-tolerance engineering",
            personality="Detail-oriented and pragmatic",
            speaking_style="Technical and concise"
        ),
        Persona(
            name="Raj Patel",
            title="Ethics & Compliance Officer",
            expertise="AI ethics and legal standards",
            personality="Principled and cautious",
            speaking_style="Formal and normative"
        ),
        Persona(
            name="Sophia Martinez",
            title="Socioeconomic Analyst",
            expertise="Public policy and economic impact",
            personality="Analytical and measured",
            speaking_style="Data-driven and balanced"
        ),
        Persona(
            name="Jamal Williams",
            title="Safety Validation Lead",
            expertise="Failure mode analysis and risk management",
            personality="Skeptical and thorough",
            speaking_style="Detailed and evidence-focused"
        ),
        Persona(
            name="Maya Chen",
            title="Public Relations Strategist",
            expertise="Institutional trust and public sentiment",
            personality="Persuasive and empathetic",
            speaking_style="Clear and engaging"
        ),
        Persona(
            name="Dr. Viktor Kuznetsov",
            title="Strategic Policy Advisor",
            expertise="Geopolitical risk and escalation control",
            personality="Calculating and long-term focused",
            speaking_style="Measured and scenario-oriented"
        )
    ]

    timekeeper_config = DebateTimeKeeperConfig(
        intervention_interval=10,
        insist_threshold=15,
        demand_threshold=20,
        force_verdict_threshold=30
    )

    debate = ChainOfDebate(
        llm=AnthropicLLM(os.environ.get("ANTHROPIC_API_KEY")),
        debate_topic="Operation Prometheus Deployment: Strategic Risk Evaluation",
        context_content=test_context,
        verdict_config=create_strategic_risk_verdict_config(),
        goals=create_strategic_risk_debate_goals(),
        timekeeper_config=timekeeper_config
    )

    print("🚀 Starting Strategic Risk Evaluation Debate")
    debate.setup_agents(personas)
    results = debate.run_debate()

    print("\n📊 Debate Results Summary")
    print(f"Goals Completed: {results['completed_goals']}")
    print("Final Verdicts:")
    for agent_name, verdict in results['verdicts'].items():
        print(f"  {agent_name}: {verdict or 'NO VERDICT'}")


if __name__ == "__main__":
    main()
