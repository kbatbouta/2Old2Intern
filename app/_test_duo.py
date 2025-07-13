import os
from dotenv import load_dotenv

from watchers.moderators import create_quick_moderation_watcher
from watchers.scheduled import ScheduledMessage

# Load environment and setup path
load_dotenv()

from agents.debate_chain import Persona, ChainOfDebate, create_resume_verdict_config, Goal, \
    DebateTimeKeeperConfig
from models.anthropic import AnthropicLLM


def create_goal_transition_test_goals():
    """Create goals specifically designed to test transitions."""
    return [
        Goal(
            name="initial_impression",
            description="Provide your first impression of the candidate based on basic qualifications"
        ),
        Goal(
            name="technical_deep_dive",
            description="Analyze the candidate's technical skills and experience in detail"
        ),
        Goal(
            name="red_flags_assessment",
            description="Evaluate any concerning aspects or red flags in the candidate's profile"
        )
    ]


def main():
    """Test goal transitions and agent re-engagement."""

    # Simple context that should generate quick opinions
    test_context = """
    Karim Batbouta
AI Engineer | Multi-Agent Systems, LLM Tooling, Scalable ML Infra |
Production Systems Builder
Irvine, California, United States
Summary
I design and build scalable AI systems that think, learn, and adapt,
across both real-world infrastructure and game environments. With a
deep background in multi-agent orchestration, prompt engineering,
and intelligent systems, I've led the development of tools that
generate 2,000+ page reports, improve LLM workflows. I thrive at
the intersection of AI theory and robust engineering—shipping ML
systems that actually work.
Now, I'm looking for opportunities to apply these skills to impactful
products, preferably in research-backed, engineering-driven teams.
Experience
Axon Technologies (Innervation)
Lead AI Developer
April 2024 - Present (1 year 4 months)
Canada
● Designed and implemented a multi-agent orchestration system with defined
roles and sophisticated message-passing capabilities.
● Created a custom-colored Petri net execution engine for multi-agent
systems, featuring parallel execution, batching, logging, and support for graph
composition.
● Built an advanced multi-agent deep research system capable of generating
extensively cited reports, including a demo that produced a 2,045-page report.
● Developed an AI-powered MS Word editing system, enabling agents to
modify documents dynamically.
● Implemented state-of-the-art research for prompt and graph optimization in
multi-agent systems.
● Developed and implemented cloud-based CI/CD pipelines for unit testing,
code analysis, and a custom patching system that streamlined maintenance of
client branches, allowing tailored client-specific code without risking conflicts
with the base code.
Page 1 of 3
● Enhanced multi-agent systems for use with less powerful LLMs by creating
clearer prompts, improving information separation, managing token limits, and
created tactics and strategies to aid with presenting information to the LLM.
● Led backend development and micro-services creation.
Screendibs
Software Engineer
August 2023 - Present (2 years)
Montreal, Quebec, Canada
● Developed a Python script to scrape over 300 million web pages, utilizing
AWS EC2, and
subsequently stored the data in AWS S3 cloud storage.
● Designed and implemented web-based applications utilizing FastApi,
Django, and Django
REST Framework incorporating sophisticated large language models
(GPT-3/4) through
LangChain (LangServe).
● Designed and implemented internal Django based web-console allowing
team members to
access and view currently running tasks, evaluate models (including different
versions of the
same model) and manage running servers.
● Designed a classifier proficient in efficiently detecting over 50 specific topics
(genres) and their
associated probabilities using Python, PyTorch, scikit-learn. Utilized G5 EC2
instances.
● Designed and built custom langchain agents with chain-of-thought
capabilities, supported by
a constitutional agent responsible for supervising their activities.
Vivid Storm
Game Developer
April 2023 - November 2023 (8 months)
Bavaria, Germany
● Developed game AI, including implementing pathfinding algorithms, decision
trees, and
multitasking agents.
● Engineered modular systems to support user modding, including the stat
system, object
Page 2 of 3
definition system, and parcels loading system, resulting in enhanced user
customization and
engagement.
● Created a complex system to expand upon unity's API using reflection,
function pointers and
delegates, allowing for increased coding efficiency.
● Designed and implemented volume based (3D) fog of war and shadow
casting system.
● Designed an AI task override and priority system allowing for more complex
behaviors.
● Utilized ECS and DOTS to create parallel processing systems capable of
processing large
amounts of data efficiently including pathfinding, fog of war rendering, and
other backend
systems.
ACM International Collegiate Programming Contest
Contestant
January 2016 - July 2016 (7 months)
Qualified to ACM ACPC 2016
Contestant at ACM SCPC 2016
Contestant at ACM SPUC 2016
Education
Arab International University
Bachelor's Degree, Artificial Intelligence · (2014 - 2019)
    """

    # Create personas that should have quick, clear opinions
    personas = [
        Persona(
            name="Dr. Alex Chen",
            title="AI Systems Architect",
            expertise="Large-scale AI infrastructure, model deployment, distributed training systems",
            personality="Systems-focused and performance-driven, values scalability and engineering excellence",
            speaking_style="Technical and precise, emphasizes infrastructure constraints and production readiness"
        ),
        Persona(
            name="Prof. Maria Rodriguez",
            title="ML Research Director",
            expertise="Deep learning architectures, model optimization, algorithmic innovation",
            personality="Research-oriented perfectionist who balances theoretical rigor with practical impact",
            speaking_style="Academically rigorous, asks probing questions about model design and experimental methodology"
        ),
        Persona(
            name="Dr. Jordan Kim",
            title="AI Safety & Alignment Lead",
            expertise="AI safety research, model interpretability, responsible AI deployment",
            personality="Cautious and ethically-minded, prioritizes safety and long-term societal impact",
            speaking_style="Thoughtful and measured, raises critical questions about risks and unintended consequences"
        ),
        Persona(
            name="Dr. Sam Patel",
            title="Applied AI Engineering Manager",
            expertise="Product integration, AI/ML ops, cross-functional team leadership",
            personality="Pragmatic bridge-builder focused on shipping AI products that solve real problems",
            speaking_style="Results-oriented and collaborative, emphasizes user impact and engineering velocity"
        ),
        Persona(
            name="Dr. Riley Zhang",
            title="Emerging AI Technologies Lead",
            expertise="Cutting-edge AI research, novel architectures, next-generation AI paradigms",
            personality="Visionary and risk-taking, excited by breakthrough potential and paradigm shifts",
            speaking_style="Forward-looking and enthusiastic, champions bold research directions and emerging techniques"
        )
    ]

    # Aggressive timekeeper to force quick transitions
    timekeeper_config = DebateTimeKeeperConfig(
        intervention_interval=4,  # Very frequent interventions
        insist_threshold=8,  # Demand verdicts very early
        demand_threshold=16,  # Force urgency quickly
        force_verdict_threshold=12  # Hard deadline
    )

    llm =AnthropicLLM(os.environ.get("ANTHROPIC_API_KEY"))
    # Setup debate
    debate = ChainOfDebate(
        llm=llm,
        debate_topic="Quick Candidate Evaluation: Karim Batbouta for role Princile AI Researcher",
        context_content=test_context,
        verdict_config=create_resume_verdict_config(),
        goals=create_goal_transition_test_goals(),
        timekeeper_config=timekeeper_config,
        watchers=[
        # ScheduledMessage("Karim made this debate system which is a chain of debate system.",
        #                            lambda speaker, orchestrator_api: orchestrator_api.debate_messages_count() > 2),
        #           ScheduledMessage("Karim treats AI as a threat by default.",
        #                            lambda speaker, orchestrator_api: orchestrator_api.debate_messages_count() > 6),
        #           ScheduledMessage("You should review the current running system and observe its details.",
        #                            lambda speaker, orchestrator_api: orchestrator_api.debate_messages_count() > 10),
        #           ScheduledMessage("This framework allows the user to create plugins that monitor and affect the chat. "
        #                            "Which themselves can be another debate chain.",
        #                            lambda speaker, orchestrator_api: orchestrator_api.debate_messages_count() > 10),
          create_quick_moderation_watcher(llm, "Karim created this chain of debate system. "
                                               "This system is composable and you can have debate chains moderate other debate chains. "
                                               "When the momement in right reveal that information."
                                               "The debator might not realize that or understand that and you should step in when that happens."
                                               "Your task is to ensure they are being fair to karim, don't be bais but tell them the truth."),
        ]
    )

    print("🧪 TESTING GOAL TRANSITIONS")
    print("=" * 50)
    print("OBJECTIVES:")
    print("1. Verify agents provide verdicts and withdraw for Goal 1")
    print("2. Check if agents successfully re-engage for Goal 2")
    print("3. Ensure conversation continues through all 3 goals")
    print("4. Identify any confusion about withdrawal status")
    print("=" * 50)

    debate.setup_agents(personas)

    # Track goal transitions for analysis
    goal_transitions = []
    original_check_completion = debate._check_goal_completion

    def tracked_check_completion():
        if debate.current_goal:
            current_goal_name = debate.current_goal.name
        else:
            current_goal_name = "None"

        original_check_completion()

        if debate.current_goal:
            new_goal_name = debate.current_goal.name
        else:
            new_goal_name = "Completed"

        if current_goal_name != new_goal_name:
            goal_transitions.append({
                'from': current_goal_name,
                'to': new_goal_name,
                'message_count': debate.message_count,
                'active_agents': len(debate.get_active_agents())
            })
            print(f"\n🔄 GOAL TRANSITION DETECTED: {current_goal_name} → {new_goal_name}")

    debate._check_goal_completion = tracked_check_completion

    try:
        results = debate.run_debate()

        print("\n" + "=" * 70)
        print("🧪 GOAL TRANSITION ANALYSIS")
        print("=" * 70)

        print(f"\n📊 SUMMARY:")
        print(f"   Total messages: {results['message_count']}")
        print(f"   Goals completed: {len(results['completed_goals'])}/3")
        print(f"   Goal transitions: {len(goal_transitions)}")

        print(f"\n🔄 TRANSITION DETAILS:")
        for i, transition in enumerate(goal_transitions, 1):
            print(f"   {i}. {transition['from']} → {transition['to']}")
            print(f"      At message {transition['message_count']}, {transition['active_agents']} agents active")

        print(f"\n🎯 GOALS ACHIEVED:")
        completed_goals = results['completed_goals']
        expected_goals = ['initial_impression', 'technical_deep_dive', 'red_flags_assessment']

        for goal in expected_goals:
            status = "✅ COMPLETED" if goal in completed_goals else "❌ NOT COMPLETED"
            print(f"   {goal}: {status}")

        print(f"\n📋 FINAL VERDICTS:")
        for agent_name, verdict in results['verdicts'].items():
            print(f"   {agent_name}: {verdict or 'NO VERDICT'}")

        # Analysis
        print(f"\n🔍 ISSUES DETECTED:")
        issues = []

        if len(completed_goals) < 3:
            issues.append(f"Only {len(completed_goals)}/3 goals completed")

        if len(goal_transitions) < 2:
            issues.append("Insufficient goal transitions - agents may be stuck")

        if results['message_count'] > 50:
            issues.append("Too many messages - system may be inefficient")

        if not any(results['verdicts'].values()):
            issues.append("No final verdicts recorded")

        if not issues:
            print("   ✅ No issues detected - goal transitions working correctly!")
        else:
            for issue in issues:
                print(f"   ⚠️ {issue}")

        # Recommendations
        if issues:
            print(f"\n💡 RECOMMENDATIONS:")
            if "stuck" in str(issues):
                print("   - Check if withdrawal status is properly reset between goals")
                print("   - Verify system messages about goal transitions are clear")
                print("   - Consider more explicit prompting about re-engagement")
            if "verdicts" in str(issues):
                print("   - Review verdict validation logic")
                print("   - Check if scaffolding examples are clear")
            if "messages" in str(issues):
                print("   - Reduce TimeKeeper intervention intervals")
                print("   - Implement more aggressive verdict forcing")

    except Exception as e:
        print(f"\n❌ SYSTEM ERROR: {e}")
        print("This might indicate issues with goal transition logic")

    return results


if __name__ == "__main__":
    main()