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
# Chain of Debate System Summary

## Overview
Chain of Debate is a multi-agent AI orchestration system designed to simulate structured expert panel discussions. The system enables multiple AI agents with distinct specializations to engage in organized debates and decision-making processes.

## Core Architecture

### Communication Layers
The system implements three levels of communication:
- **Public messages** visible to all participants
- **Private whisper messages** between specific agents
- **Internal thoughts** visible only to the generating agent

### Sequential Processing
Complex problems are broken down into sequential goals that are addressed systematically, with automatic progression between discussion phases and built-in moderation to maintain focus.

## Key Technical Features

### Message Structure
All communications follow a standardized XML-like format with structured verdict tracking, reasoning requirements, and participation status management.

### Agent Coordination
- Private communication channels enable strategic coordination
- Coalition formation detection and analysis
- Enhanced whisper engagement mechanisms to encourage private discussions
- Information filtering based on message visibility rules

### Moderation System
An automated "TimeKeeper" agent provides escalating intervention to ensure progress, from gentle reminders to forced decision deadlines.

## Differentiation from Existing Systems

### Current AI Approaches
- Single-agent systems provide one perspective
- Simple multi-agent systems collect separate responses without interaction
- Limited coordination or strategic behavior between agents

### Chain of Debate Innovations
- Interactive multi-agent discussions with realistic group dynamics
- Private information sharing and alliance formation
- Systematic problem decomposition and progression
- Emergent strategic behavior and coalition building

## Applications
The system can be configured for various evaluation scenarios including hiring decisions, research review, investment analysis, policy assessment, and other multi-stakeholder decision-making processes.

## Development Status
Currently in active development with stable core components including orchestration engine, communication systems, and goal processing. Planned enhancements include advanced analytics, performance optimization, and expanded domain templates.

## Technical Implementation
Built on configurable persona systems with domain-specific verdict types, extensible validation rules, and comprehensive testing frameworks for multi-agent behavior validation.
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

    llm = AnthropicLLM(os.environ.get("ANTHROPIC_API_KEY"))
    llm.set_model("claude-3-5-sonnet-latest")
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
