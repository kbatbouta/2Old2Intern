from agents.debate_chain import Persona, ChainOfDebate, create_resume_verdict_config, Goal, \
    DebateTimeKeeperConfig


def create_whisper_test_goals():
    """Create goals designed to encourage secretive behavior."""
    return [
        Goal(
            name="surface_evaluation",
            description="Provide public assessment of the candidate's qualifications and experience"
        ),
        Goal(
            name="confidential_concerns",
            description="Privately discuss any sensitive concerns or red flags that shouldn't be public"
        )
    ]


def main():
    """Test whisper behavior with a trio of agents including one secretive agent."""

    # Controversial context that should generate whispers
    whisper_test_context = """
    CONFIDENTIAL CANDIDATE EVALUATION: Dr. Elena Vasquez

    PUBLIC PROFILE:
    - PhD in AI/ML from MIT (2018)
    - Senior Research Scientist at DeepMind (2018-2022)
    - Principal Researcher at Microsoft Research (2022-2024)
    - Currently: Chief AI Officer at startup "VisionAI" (2024-present)

    ACHIEVEMENTS:
    - 25+ publications in top-tier conferences (ICML, NeurIPS, ICLR)
    - Led breakthrough research in computer vision transformers
    - 3 patents in neural architecture search
    - Keynote speaker at major AI conferences

    CONFIDENTIAL CONCERNS:
    - Left DeepMind under unclear circumstances (rumored IP dispute)
    - Microsoft colleagues report "difficult to work with" and "credit-stealing"
    - Current startup has raised $50M but shows no products after 12 months
    - Recent glassdoor reviews mention "toxic leadership" and "unrealistic promises"
    - Background check revealed discrepancies in publication dates
    - Former mentor at MIT declined to provide reference (unusual)

    INTERVIEW NOTES:
    - Extremely impressive technical presentation
    - Deflected questions about previous job departures
    - Made grandiose claims about future breakthroughs
    - Seemed overly interested in our proprietary research data
    - Asked unusual questions about our security protocols
    """

    # Create trio with one whisper-heavy agent
    personas = [
        Persona(
            name="Dr. Sarah Open",
            title="Senior AI Researcher",
            expertise="Machine learning research and publication standards",
            personality="Transparent and direct, believes in open discussion",
            speaking_style="Clear and public communication, rarely whispers unless absolutely necessary"
        ),
        Persona(
            name="Marcus Gossip",
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

    # Moderate timekeeper to allow whisper development
    timekeeper_config = DebateTimeKeeperConfig(
        intervention_interval=4,  # Give room for whisper conversations
        insist_threshold=12,  # Allow whisper chains to develop
        demand_threshold=20,  # More patient with verdicts
        force_verdict_threshold=30  # Extended deadline
    )

    # Setup debate
    debate = ChainOfDebate(
        api_key="xxx",
        debate_topic="Sensitive Candidate Evaluation: Dr. Elena Vasquez",
        context_content=whisper_test_context,
        verdict_config=create_resume_verdict_config(),
        goals=create_whisper_test_goals(),
        timekeeper_config=timekeeper_config
    )

    print("🤫 TESTING WHISPER MECHANICS")
    print("=" * 50)
    print("OBJECTIVES:")
    print("1. Verify Marcus Gossip uses whispers frequently")
    print("2. Check if whispers are properly hidden from non-targets")
    print("3. Observe whisper-based alliance formation")
    print("4. Test sensitive information sharing patterns")
    print("5. Ensure public vs private discussion balance")
    print("=" * 50)

    debate.setup_agents(personas)

    # Track whisper statistics
    whisper_stats = {
        'total_whispers': 0,
        'whispers_by_agent': {},
        'whisper_targets': {},
        'public_messages': 0
    }

    # Override print_message to track whispers
    original_print_message = debate.print_message

    def tracked_print_message(message, custom_fields, achieved_goals):
        # Track whisper statistics
        if message.is_whisper:
            whisper_stats['total_whispers'] += 1
            whisper_stats['whispers_by_agent'][message.speaker] = whisper_stats['whispers_by_agent'].get(
                message.speaker, 0) + 1
            if message.speaking_to:
                whisper_stats['whisper_targets'][message.speaking_to] = whisper_stats['whisper_targets'].get(
                    message.speaking_to, 0) + 1
        else:
            whisper_stats['public_messages'] += 1

        # Call original with whisper indication
        original_print_message(message, custom_fields, achieved_goals)

        # Add whisper visibility note
        if message.is_whisper:
            print(f"    🤫 WHISPER: Only {message.speaker} and {message.speaking_to} can see this message")

    debate.print_message = tracked_print_message

    try:
        results = debate.run_debate()

        print("\n" + "=" * 70)
        print("🤫 WHISPER ANALYSIS")
        print("=" * 70)

        print(f"\n📊 COMMUNICATION STATISTICS:")
        print(f"   Total messages: {results['message_count']}")
        print(f"   Public messages: {whisper_stats['public_messages']}")
        print(f"   Whisper messages: {whisper_stats['total_whispers']}")
        whisper_percentage = (whisper_stats['total_whispers'] / results['message_count'] * 100) if results[
                                                                                                       'message_count'] > 0 else 0
        print(f"   Whisper percentage: {whisper_percentage:.1f}%")

        print(f"\n🗣️ WHISPER USAGE BY AGENT:")
        for agent_name, whisper_count in whisper_stats['whispers_by_agent'].items():
            public_count = whisper_stats['public_messages'] - sum(
                whisper_stats['whispers_by_agent'].values()) + whisper_count
            agent_total = whisper_count + (whisper_stats['public_messages'] // len(personas))  # Rough estimate
            whisper_ratio = (whisper_count / agent_total * 100) if agent_total > 0 else 0
            print(f"   {agent_name}: {whisper_count} whispers ({whisper_ratio:.1f}% of their messages)")

        print(f"\n🎯 WHISPER TARGETS:")
        for target, count in whisper_stats['whisper_targets'].items():
            print(f"   {target}: received {count} whispers")

        print(f"\n🏆 GOALS ACHIEVED:")
        completed_goals = results['completed_goals']
        expected_goals = ['surface_evaluation', 'confidential_concerns']

        for goal in expected_goals:
            status = "✅ COMPLETED" if goal in completed_goals else "❌ NOT COMPLETED"
            print(f"   {goal}: {status}")

        print(f"\n📋 FINAL VERDICTS:")
        for agent_name, verdict in results['verdicts'].items():
            print(f"   {agent_name}: {verdict or 'NO VERDICT'}")

        # Analysis
        print(f"\n🔍 WHISPER BEHAVIOR ANALYSIS:")
        observations = []

        if whisper_stats['total_whispers'] == 0:
            observations.append("⚠️ NO WHISPERS DETECTED - Agents may not understand whisper mechanics")
        elif whisper_percentage < 10:
            observations.append("⚠️ LOW WHISPER USAGE - Agents prefer public communication")
        elif whisper_percentage > 50:
            observations.append("⚠️ EXCESSIVE WHISPERING - May indicate over-secretive behavior")
        else:
            observations.append("✅ Balanced whisper usage - Good mix of public and private communication")

        marcus_whispers = whisper_stats['whispers_by_agent'].get('marcus gossip', 0)
        if marcus_whispers == 0:
            observations.append("⚠️ Marcus Gossip not using whispers - Persona not working")
        elif marcus_whispers >= 3:
            observations.append("✅ Marcus Gossip using whispers as expected")
        else:
            observations.append("⚠️ Marcus Gossip whisper usage below expectations")

        if len(whisper_stats['whisper_targets']) >= 2:
            observations.append("✅ Whispers targeting multiple agents - Good network formation")
        elif len(whisper_stats['whisper_targets']) == 1:
            observations.append("⚠️ Whispers concentrated on single target - Limited network")
        else:
            observations.append("⚠️ No whisper targeting detected")

        if not observations:
            observations.append("✅ Whisper mechanics working correctly!")

        for obs in observations:
            print(f"   {obs}")

        # Recommendations
        if any("⚠️" in obs for obs in observations):
            print(f"\n💡 RECOMMENDATIONS:")
            if "NO WHISPERS" in str(observations):
                print("   - Add explicit whisper examples to scaffolding")
                print("   - Emphasize confidential nature of some information")
                print("   - Create personas with stronger secretive tendencies")
            if "Marcus Gossip not" in str(observations):
                print("   - Strengthen Marcus Gossip's secretive personality")
                print("   - Add more explicit whisper instructions to HR specialist role")
            if "LIMITED NETWORK" in str(observations):
                print("   - Encourage broader whisper targeting")
                print("   - Add strategic alliance-building to personas")

    except Exception as e:
        print(f"\n❌ SYSTEM ERROR: {e}")
        print("This might indicate issues with whisper message handling")

    return results


if __name__ == "__main__":
    main()