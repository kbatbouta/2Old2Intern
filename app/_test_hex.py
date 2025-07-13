import os
from dotenv import load_dotenv

# Load environment and setup path
load_dotenv()

from agents.debate_chain import Persona, ChainOfDebate, VerdictConfig, Goal, DebateTimeKeeperConfig
from models.anthropic import AnthropicLLM


def create_climate_verdict_config():
    """Create verdict configuration for climate policy evaluation."""
    return VerdictConfig(
        verdict_options=["SUPPORT", "SUPPORT_WITH_CONDITIONS", "OPPOSE", "ABSTAIN"],
        verdict_descriptions={
            "SUPPORT": "Fully endorse the proposal as written",
            "SUPPORT_WITH_CONDITIONS": "Support with specific modifications required",
            "OPPOSE": "Reject the proposal entirely",
            "ABSTAIN": "Cannot reach a position due to conflicting priorities"
        }
    )


def create_climate_goals():
    """Create goals that encourage coalition building and conflicts."""
    return [
        Goal(
            name="economic_impact",
            description="Assess the economic costs and benefits of the carbon tax proposal"
        ),
        Goal(
            name="political_feasibility",
            description="Evaluate the political viability and public acceptance challenges"
        ),
        Goal(
            name="final_recommendation",
            description="Reach consensus or document disagreements on the proposal"
        )
    ]


def main():
    """6-agent debate on carbon tax with built-in conflicts and alliance potential."""

    # Controversial but real policy proposal
    climate_context = """
    PROPOSED FEDERAL CARBON TAX POLICY

    SUMMARY:
    Congress is considering a federal carbon tax starting at $50/ton CO2, rising to $150/ton by 2035.
    Revenue would fund: 50% direct rebates to households, 30% clean energy infrastructure, 20% deficit reduction.

    KEY PROVISIONS:
    - Applies to fossil fuel producers/importers at point of extraction/import
    - Border carbon adjustments to protect domestic industry
    - Exemptions for agricultural fuels and small emitters (<25,000 tons/year)
    - Automatic adjustment mechanism tied to emission reduction targets
    - Sunset clause: policy expires in 2040 unless reauthorized

    ECONOMIC PROJECTIONS:
    - Estimated $200B annual revenue at full implementation
    - 40-60% reduction in US emissions by 2035 (vs 2005 baseline)
    - GDP impact: -0.1% to +0.3% depending on recycling mechanism
    - Job losses in fossil fuel sectors: ~200,000 over 10 years
    - Job gains in clean energy: ~400,000 over 10 years

    POLITICAL CONTEXT:
    - Support: Environmental groups, some economists, EU pressure for climate action
    - Opposition: Oil/gas industry, coal states, taxpayer advocates, some unions
    - Public polling: 45% support, 35% oppose, 20% undecided
    - State variations: Blue states 60% support, red states 25% support

    IMPLEMENTATION CHALLENGES:
    - Supreme Court may review federal taxation authority
    - Industry threatens to relocate to avoid tax
    - Rural communities disproportionately affected by energy cost increases
    - Coordination required with state-level climate policies
    """

    # 6 agents with natural conflicts and alliance potential
    personas = [
        Persona(
            name="Dr. Rachel Green",
            title="Environmental Policy Director",
            expertise="Climate science, environmental regulation, sustainability policy",
            personality="Passionate about climate action, data-driven, impatient with economic excuses, forms alliances with other pro-environment voices",
            speaking_style="Uses scientific evidence aggressively, whispers strategy with allies, dismissive of fossil fuel industry concerns"
        ),

        Persona(
            name="Marcus Steel",
            title="Energy Industry Lobbyist",
            expertise="Fossil fuel economics, energy markets, regulatory compliance costs",
            personality="Protective of industry interests, skilled at finding policy flaws, builds coalitions with economic conservatives, secretive about industry talking points",
            speaking_style="Focuses on economic damage and job losses, whispers confidential industry data, questions scientific projections"
        ),

        Persona(
            name="Senator Patricia Webb",
            title="Swing State Senator",
            expertise="Political strategy, electoral consequences, legislative process",
            personality="Pragmatic politician balancing environmental and economic concerns, seeks middle ground, privately shares electoral calculations",
            speaking_style="Diplomatic in public but reveals true political concerns in whispers, focuses on voter reactions and reelection"
        ),

        Persona(
            name="Dr. James Brooks",
            title="Economic Policy Analyst",
            expertise="Tax policy, macroeconomic modeling, fiscal impact analysis",
            personality="Numbers-focused technocrat, skeptical of both environmental activism and industry scare tactics, shares technical details privately",
            speaking_style="Dry technical analysis publicly, whispers about data limitations and modeling assumptions with fellow wonks"
        ),

        Persona(
            name="Maria Santos",
            title="Labor Union Representative",
            expertise="Worker impacts, just transition policies, industrial relations",
            personality="Protective of union jobs but worried about climate impacts on workers, torn between environmental and economic priorities, builds coalitions across class lines",
            speaking_style="Emotional about worker impacts, whispers about internal union divisions, seeks alliances with both greens and industry when worker interests align"
        ),

        Persona(
            name="Robert Chase",
            title="Fiscal Conservative Think Tank Fellow",
            expertise="Government spending, taxation policy, regulatory burden analysis",
            personality="Ideologically opposed to new taxes and government intervention, forms alliances with industry against big government, shares libertarian talking points privately",
            speaking_style="Attacks government overreach and spending, whispers about political strategy with industry allies, questions government competence"
        )
    ]

    # Extended timekeeper for complex alliances
    timekeeper_config = DebateTimeKeeperConfig(
        intervention_interval=5,  # More room for whisper networks
        insist_threshold=20,  # Allow coalition building
        demand_threshold=35,  # Extended debate time
        force_verdict_threshold=50  # Lengthy deadline for complex issue
    )

    # Setup debate
    debate = ChainOfDebate(
        llm=AnthropicLLM(os.environ.get("ANTHROPIC_API_KEY")),
        debate_topic="Federal Carbon Tax Policy Proposal",
        context_content=climate_context,
        verdict_config=create_climate_verdict_config(),
        goals=create_climate_goals(),
        timekeeper_config=timekeeper_config
    )

    print("🏛️ CLIMATE POLICY DEBATE - 6 AGENTS")
    print("=" * 60)
    print("EXPECTED CONFLICTS:")
    print("• Environment (Green) vs Industry (Steel)")
    print("• Labor (Santos) torn between jobs and climate")
    print("• Fiscal Conservative (Chase) vs any new taxes")
    print("• Senator (Webb) balancing political pressures")
    print("• Economist (Brooks) providing technical reality checks")
    print("\nEXPECTED ALLIANCES:")
    print("• Green + Santos (environmental justice)")
    print("• Steel + Chase (anti-tax coalition)")
    print("• Webb + Brooks (pragmatic center)")
    print("• Cross-cutting alliances on specific provisions")
    print("=" * 60)

    debate.setup_agents(personas)

    # Enhanced whisper tracking
    whisper_stats = {
        'total_whispers': 0,
        'whispers_by_agent': {},
        'whisper_networks': {},  # Track who whispers to whom
        'alliance_formations': [],  # Track coalition building
        'public_messages': 0,
        'conflict_moments': 0
    }

    original_print_message = debate.print_message

    def enhanced_print_message(message, custom_fields, achieved_goals):
        # Track statistics
        if message.is_whisper:
            whisper_stats['total_whispers'] += 1
            whisper_stats['whispers_by_agent'][message.speaker] = whisper_stats['whispers_by_agent'].get(
                message.speaker, 0) + 1

            # Track whisper networks
            if message.speaker not in whisper_stats['whisper_networks']:
                whisper_stats['whisper_networks'][message.speaker] = []
            if message.speaking_to not in whisper_stats['whisper_networks'][message.speaker]:
                whisper_stats['whisper_networks'][message.speaker].append(message.speaking_to)

            # Detect alliance formation keywords
            alliance_keywords = ['alliance', 'coalition', 'together', 'coordinate', 'strategy', 'team up', 'work with']
            if any(keyword in message.content.lower() for keyword in alliance_keywords):
                whisper_stats['alliance_formations'].append(f"{message.speaker} → {message.speaking_to}")
        else:
            whisper_stats['public_messages'] += 1

        # Detect conflict moments
        conflict_keywords = ['disagree', 'wrong', 'oppose', 'reject', 'flawed', 'misguided', 'naive']
        if any(keyword in message.content.lower() for keyword in conflict_keywords):
            whisper_stats['conflict_moments'] += 1

        # Call original
        original_print_message(message, custom_fields, achieved_goals)

        # Add enhanced annotations
        if message.is_whisper:
            print(f"    🤫 PRIVATE: Only {message.speaker} and {message.speaking_to} can see this")

        # Detect coalition building
        if message.is_whisper and any(
                word in message.content.lower() for word in ['alliance', 'coordinate', 'strategy']):
            print(f"    🤝 COALITION BUILDING DETECTED")

    debate.print_message = enhanced_print_message

    try:
        results = debate.run_debate()

        # Enhanced analysis
        print("\n" + "=" * 70)
        print("🏛️ DEBATE ANALYSIS")
        print("=" * 70)

        print(f"\n📊 COMMUNICATION PATTERNS:")
        print(f"   Total messages: {results['message_count']}")
        print(f"   Public debates: {whisper_stats['public_messages']}")
        print(f"   Private whispers: {whisper_stats['total_whispers']}")
        print(f"   Conflict moments: {whisper_stats['conflict_moments']}")

        print(f"\n🗣️ WHISPER ACTIVITY:")
        for agent, count in sorted(whisper_stats['whispers_by_agent'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {agent}: {count} private communications")

        print(f"\n🕸️ WHISPER NETWORKS:")
        for agent, targets in whisper_stats['whisper_networks'].items():
            print(f"   {agent} → {', '.join(targets)}")

        if whisper_stats['alliance_formations']:
            print(f"\n🤝 COALITION BUILDING DETECTED:")
            for formation in whisper_stats['alliance_formations']:
                print(f"   {formation}")
        else:
            print(f"\n🤝 No explicit coalition building detected in whispers")

        print(f"\n📋 FINAL POSITIONS:")
        for agent, verdict in results['verdicts'].items():
            print(f"   {agent}: {verdict or 'NO POSITION'}")

        # Identify winning coalitions
        verdict_counts = {}
        for verdict in results['verdicts'].values():
            if verdict:
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        print(f"\n🏆 COALITION OUTCOMES:")
        for verdict, count in sorted(verdict_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {verdict}: {count} agents ({count / 6 * 100:.1f}%)")

    except Exception as e:
        print(f"\n❌ DEBATE ERROR: {e}")
        import traceback
        traceback.print_exc()

    return results


if __name__ == "__main__":
    main()