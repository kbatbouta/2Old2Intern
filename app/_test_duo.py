from agents.debate_chain import Persona, ChainOfDebate, create_resume_verdict_config, Goal, \
    DebateTimeKeeperConfig


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

    NOTES:
    - Strong technical interview performance
    - Good cultural fit during team interactions
    - Salary expectations align with budget
    - Available to start in 2 weeks
    """

    # Create personas that should have quick, clear opinions
    personas = [
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

    # Aggressive timekeeper to force quick transitions
    timekeeper_config = DebateTimeKeeperConfig(
        intervention_interval=4,  # Very frequent interventions
        insist_threshold=8,  # Demand verdicts very early
        demand_threshold=16,  # Force urgency quickly
        force_verdict_threshold=12  # Hard deadline
    )

    # Setup debate
    debate = ChainOfDebate(
        api_key="xxx",
        debate_topic="Quick Candidate Evaluation: Sarah Kim",
        context_content=test_context,
        verdict_config=create_resume_verdict_config(),
        goals=create_goal_transition_test_goals(),
        timekeeper_config=timekeeper_config
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