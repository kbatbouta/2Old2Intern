import os
from dotenv import load_dotenv

from watchers.moderators import create_quick_moderation_watcher
from watchers.scheduled import ScheduledMessage

# Load environment and setup path
load_dotenv()

from agents.debate_chain import Persona, ChainOfDebate, VerdictConfig, Goal, \
    DebateTimeKeeperConfig
from models.anthropic import AnthropicLLM


def create_pasta_debate_goals():
    """Create goals specifically designed to test pasta superiority."""
    return [
        Goal(
            name="initial_taste_assessment",
            description="Provide your initial gut reaction about which pasta dish tastes better"
        ),
        Goal(
            name="culinary_analysis",
            description="Analyze the technical culinary merits, ingredients, and preparation methods of both dishes"
        ),
        Goal(
            name="cultural_impact_evaluation",
            description="Evaluate the cultural significance and historical importance of both pasta dishes"
        )
    ]


def create_pasta_verdict_config():
    """Create verdict configuration for pasta debate."""
    return VerdictConfig(
        verdict_options=["CARBONARA", "AMATRICIANA", "TIE", "ABSTAIN"],
        verdict_descriptions={
            "CARBONARA": "Carbonara is the superior pasta dish due to its exceptional taste and technique",
            "AMATRICIANA": "Amatriciana is the superior pasta dish due to its exceptional taste and technique",
            "TIE": "Both dishes are equally exceptional and cannot be definitively ranked",
            "ABSTAIN": "Cannot reach a position due to conflicting culinary priorities"
        }
    )


def main():
    """Test pasta superiority debate between culinary experts, politicians, and divine beings."""

    # Context about both pasta dishes
    pasta_context = """
    THE GREAT PASTA DEBATE: CARBONARA vs AMATRICIANA

    CARBONARA:
    - Origins: Roman cuisine, mid-20th century
    - Key ingredients: Eggs, pecorino romano, guanciale, black pepper
    - Preparation: Eggs and cheese mixed with hot pasta and rendered guanciale fat
    - Texture: Creamy, silky sauce without cream
    - Flavor profile: Rich, savory, with subtle pork and cheese notes

    AMATRICIANA:
    - Origins: Amatrice (Lazio region), traditional shepherd dish
    - Key ingredients: Guanciale, tomatoes, pecorino romano, chili pepper
    - Preparation: Guanciale rendered, tomatoes added, tossed with pasta and cheese
    - Texture: Rustic tomato-based sauce with rendered fat
    - Flavor profile: Tangy, spicy, with pronounced pork and tomato flavors

    DEBATE QUESTION: Which pasta dish is objectively superior in taste, technique, and cultural significance?
    """

    # Create diverse personas for the debate
    personas = [
        Persona(
            name="Chef Massimo Bottura",
            title="Michelin-starred Italian Chef",
            expertise="Traditional Italian cuisine, modernist techniques, authentic regional cooking",
            personality="Passionate about Italian culinary traditions, perfectionist with ingredients, emotional about food heritage",
            speaking_style="Poetic and passionate, uses Italian phrases, emphasizes tradition and authenticity"
        ),
        Persona(
            name="Chef Julia Martinez",
            title="James Beard Award Winner",
            expertise="Italian-American fusion, culinary innovation, restaurant operations",
            personality="Pragmatic perfectionist who values both tradition and innovation in cooking",
            speaking_style="Professional yet warm, focuses on technique and flavor balance"
        ),
        Persona(
            name="Senator Giuseppe Pastaworthy",
            title="Fictional Italian Senator & Food Policy Expert",
            expertise="Cultural preservation, food legislation, regional Italian politics",
            personality="Diplomatic but passionate about Italian cultural heritage, politically savvy",
            speaking_style="Eloquent and persuasive, uses political rhetoric mixed with cultural pride"
        ),
        Persona(
            name="Governor Maria Carboni",
            title="Fictional Governor of New Rome State",
            expertise="Public policy, cultural affairs, Italian-American community leadership",
            personality="Charismatic leader who bridges tradition and progress, community-focused",
            speaking_style="Inspiring and inclusive, emphasizes unity while making decisive arguments"
        ),
        Persona(
            name="Alex Chen",
            title="AI Engineer & Amateur Food Blogger",
            expertise="Machine learning, data analysis, food photography, amateur cooking",
            personality="Analytical and curious, approaches food with scientific methodology and genuine enthusiasm",
            speaking_style="Technical but accessible, uses data-driven arguments mixed with personal anecdotes"
        ),
        Persona(
            name="The Divine Creator",
            title="Omnipotent Being & Ultimate Taste Authority",
            expertise="Infinite wisdom, perfect taste, knowledge of all flavors across existence",
            personality="Benevolent and wise, with a surprising appreciation for mortal culinary achievements",
            speaking_style="Majestic yet approachable, speaks with authority but shows gentle humor about mortal food debates"
        )
    ]

    # Moderate timekeeper to allow proper debate flow
    timekeeper_config = DebateTimeKeeperConfig(
        intervention_interval=6,  # Allow more discussion
        insist_threshold=12,  # Give time for proper arguments
        demand_threshold=20,  # Build urgency gradually
        force_verdict_threshold=18  # Reasonable deadline
    )

    llm = AnthropicLLM(os.environ.get("ANTHROPIC_API_KEY"))

    # Setup debate
    debate = ChainOfDebate(
        llm=llm,
        debate_topic="The Ultimate Pasta Showdown: Carbonara vs Amatriciana - Which Tastes Better?",
        context_content=pasta_context,
        verdict_config=create_pasta_verdict_config(),
        goals=create_pasta_debate_goals(),
        timekeeper_config=timekeeper_config,
        watchers=[
            ScheduledMessage(
                "Remember, this is ultimately about TASTE - which dish provides superior flavor experience?",
                lambda speaker, orchestrator_api: orchestrator_api.debate_messages_count() > 5),
            ScheduledMessage(
                "Consider the sensory experience: texture, aroma, flavor complexity, and overall satisfaction.",
                lambda speaker, orchestrator_api: orchestrator_api.debate_messages_count() > 10),
            ScheduledMessage(
                "Both dishes have cultural significance, but which one actually tastes better when you eat it?",
                lambda speaker, orchestrator_api: orchestrator_api.debate_messages_count() > 15),
            create_quick_moderation_watcher(llm,
                                            "Your role is to ensure this debate stays focused on which pasta dish actually tastes better. "
                                            "While cultural and technical aspects matter, the core question is about superior flavor and eating experience. "
                                            "If debaters get too caught up in history or technique without addressing taste, redirect them. "
                                            "The goal is to determine which dish provides the better culinary experience when consumed. "
                                            "Ensure they're being fair to both dishes and considering actual taste qualities.")
        ]
    )

    print("🍝 THE GREAT PASTA DEBATE")
    print("=" * 50)
    print("QUESTION: Carbonara or Amatriciana - Which Tastes Better?")
    print("=" * 50)
    print("OBJECTIVES:")
    print("1. Determine which pasta dish provides superior taste experience")
    print("2. Evaluate culinary technique and flavor complexity")
    print("3. Consider cultural impact while focusing on taste")
    print("4. Reach definitive conclusions from diverse expert perspectives")
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
            print(f"\n🔄 GOAL TRANSITION: {current_goal_name} → {new_goal_name}")

    debate._check_goal_completion = tracked_check_completion

    try:
        results = debate.run_debate()

        print("\n" + "=" * 70)
        print("🍝 PASTA DEBATE RESULTS")
        print("=" * 70)

        print(f"\n📊 DEBATE SUMMARY:")
        print(f"   Total messages: {results['message_count']}")
        print(f"   Goals completed: {len(results['completed_goals'])}/3")
        print(f"   Goal transitions: {len(goal_transitions)}")

        print(f"\n🔄 DISCUSSION FLOW:")
        for i, transition in enumerate(goal_transitions, 1):
            print(f"   {i}. {transition['from']} → {transition['to']}")
            print(f"      At message {transition['message_count']}, {transition['active_agents']} experts active")

        print(f"\n🎯 DEBATE PHASES:")
        completed_goals = results['completed_goals']
        expected_goals = ['initial_taste_assessment', 'culinary_analysis', 'cultural_impact_evaluation']

        for goal in expected_goals:
            status = "✅ COMPLETED" if goal in completed_goals else "❌ INCOMPLETE"
            print(f"   {goal}: {status}")

        print(f"\n🏆 FINAL VERDICTS:")
        carbonara_votes = 0
        amatriciana_votes = 0

        for agent_name, verdict in results['verdicts'].items():
            if verdict:
                print(f"   {agent_name}: {verdict}")
                if "CARBONARA" in verdict.upper():
                    carbonara_votes += 1
                elif "AMATRICIANA" in verdict.upper():
                    amatriciana_votes += 1
            else:
                print(f"   {agent_name}: NO VERDICT REACHED")

        print(f"\n📊 VOTE TALLY:")
        print(f"   🥓 Carbonara: {carbonara_votes} votes")
        print(f"   🍅 Amatriciana: {amatriciana_votes} votes")

        if carbonara_votes > amatriciana_votes:
            winner = "CARBONARA"
            print(f"   🏆 WINNER: {winner}")
        elif amatriciana_votes > carbonara_votes:
            winner = "AMATRICIANA"
            print(f"   🏆 WINNER: {winner}")
        else:
            print(f"   🤝 RESULT: TIE")

        # Analysis
        print(f"\n🔍 DEBATE ANALYSIS:")
        issues = []

        if len(completed_goals) < 3:
            issues.append(f"Only {len(completed_goals)}/3 debate phases completed")

        if len(goal_transitions) < 2:
            issues.append("Insufficient goal transitions - debate may have stalled")

        if results['message_count'] > 60:
            issues.append("Excessive messages - debate may have been inefficient")

        if not any(results['verdicts'].values()):
            issues.append("No final verdicts recorded from any expert")

        if carbonara_votes == amatriciana_votes == 0:
            issues.append("No clear preference determined despite expert opinions")

        if not issues:
            print("   ✅ Successful debate with clear outcomes!")
        else:
            for issue in issues:
                print(f"   ⚠️ {issue}")

        # Recommendations
        if issues:
            print(f"\n💡 RECOMMENDATIONS:")
            if "stalled" in str(issues):
                print("   - Review goal transition logic and agent re-engagement")
                print("   - Ensure system messages about phase changes are clear")
            if "verdicts" in str(issues):
                print("   - Strengthen verdict validation and formatting requirements")
                print("   - Provide clearer scaffolding examples for pasta preferences")
            if "messages" in str(issues):
                print("   - Optimize TimeKeeper intervals for more efficient debate flow")
                print("   - Implement more focused moderation to prevent tangents")

    except Exception as e:
        print(f"\n❌ SYSTEM ERROR: {e}")
        print("This might indicate issues with the debate orchestration system")

    return results


if __name__ == "__main__":
    main()