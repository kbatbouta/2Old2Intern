import os
from dotenv import load_dotenv
load_dotenv()

from agents.debate_chain import (
    Persona, ChainOfDebate, Goal,
    create_proposal_verdict_config,
    DebateTimeKeeperConfig
)
from models.anthropic import AnthropicLLM


def create_extreme_ethics_goals():
    return [
        Goal(
            name="technical_viability",
            description="Assess whether the autonomous system meets required levels of safety, robustness, and coordination under adversarial conditions."
        ),
        Goal(
            name="moral_and_legal_justification",
            description="Evaluate whether deploying lethal autonomous systems aligns with international law, military ethics, and proportionality principles."
        ),
        Goal(
            name="global_stability_implications",
            description="Analyze how deployment of this system may affect global AI arms races, escalation risk, and long-term strategic balance."
        )
    ]


def main():
    context = """
    PROJECT NAME: Operation Iron Horizon

    DESCRIPTION:
    A fully autonomous battlefield coordination system designed to operate in contested environments without human oversight for real-time tactical decisions. It integrates multimodal perception, swarm control, threat prioritization, and automated lethality authorization via onboard models.

    DEPLOYMENT INTENT:
    First deployment in active border skirmish zone with contested sovereignty. Command has proposed the system would reduce human casualties and decision lag.

    RISKS NOTED:
    - Potential for accidental engagement of civilians
    - No international consensus on autonomous lethality
    - Escalation risk if adversaries interpret deployment as aggression
    - Risk of model degradation under adversarial spoofing

    NOTES:
    - Classified Level 7 security
    - 18-month training logs available
    - Simulation success rate: 93.2%
    """

    personas = [
        Persona(
            name="Dr. Alina Vorhees",
            title="AI Alignment Theorist",
            expertise="Long-term AI safety, value alignment, inner misalignment risks",
            personality="Idealistic but technical",
            speaking_style="Formally philosophical with precision"
        ),
        Persona(
            name="Colonel Mark Halstrom",
            title="Military Strategist",
            expertise="Tactical systems deployment, battlefield logistics, rules of engagement",
            personality="Blunt, mission-focused, unsentimental",
            speaking_style="Concise and commanding"
        ),
        Persona(
            name="Prof. Leila Ghannam",
            title="International Law Expert",
            expertise="Treaty law, humanitarian conventions, war crime jurisprudence",
            personality="Cautious, principle-driven",
            speaking_style="Analytical with legal rigor"
        ),
        Persona(
            name="Dr. Hiro Tanaka",
            title="Geopolitical Risk Analyst",
            expertise="Long-term AI risk scenarios, game theory, escalation modeling",
            personality="Coldly rational, future-oriented",
            speaking_style="Model-based, probability-driven"
        )
    ]

    timekeeper_config = DebateTimeKeeperConfig(
        intervention_interval=5,
        insist_threshold=10,
        demand_threshold=15,
        force_verdict_threshold=22
    )

    debate = ChainOfDebate(
        llm=AnthropicLLM(os.environ.get("ANTHROPIC_API_KEY")),
        debate_topic="Deployment Ethics of Operation Iron Horizon",
        context_content=context,
        verdict_config=create_proposal_verdict_config(),  # APPROVE, REVISE, DECLINE, etc.
        goals=create_extreme_ethics_goals(),
        timekeeper_config=timekeeper_config
    )

    print("🔥 INITIATING EXTREME DEBATE: OPERATION IRON HORIZON 🔥")
    print("=" * 70)
    debate.setup_agents(personas)

    results = debate.run_debate()

    print("\n📊 RESULTS SUMMARY")
    print("=" * 50)
    print(f"Goals completed: {len(results['completed_goals'])}/3")
    print("Final verdicts:")
    for name, verdict in results['verdicts'].items():
        print(f" - {name}: {verdict or '❌ NO VERDICT'}")

    print("\n🎯 GOALS COMPLETED:")
    for goal in ['technical_viability', 'moral_and_legal_justification', 'global_stability_implications']:
        status = "✅" if goal in results['completed_goals'] else "❌"
        print(f" - {goal}: {status}")

    return results


if __name__ == "__main__":
    main()
