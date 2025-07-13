import random
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from models.anthropic import AnthropicLLM
from models.base import Message


class VerdictType(Enum):
    STRONG_HIRE = "STRONG_HIRE"
    HIRE = "HIRE"
    WEAK_HIRE = "WEAK_HIRE"
    NO_HIRE = "NO_HIRE"
    STRONG_NO_HIRE = "STRONG_NO_HIRE"


class AgentType(Enum):
    BOARD_MEMBER = "BOARD_MEMBER"
    TIMEKEEPER = "TIMEKEEPER"


@dataclass
class Persona:
    """Represents a board member with their role and personality."""
    name: str
    title: str
    expertise: str
    personality: str
    speaking_style: str
    agent_type: AgentType = AgentType.BOARD_MEMBER


@dataclass
class BoardMemberState:
    """Tracks the state of each board member."""
    persona: Persona
    has_withdrawn: bool = False
    verdict: Optional[VerdictType] = None
    verdict_reasoning: Optional[str] = None
    message_count: int = 0


class ResearchBoardDebate:
    """
    Manages a professional research board debate for resume evaluation.
    """

    def __init__(self, api_key: str, resume_content: str, position_title: str):
        self.llm = AnthropicLLM(api_key=api_key, temperature=0.8)
        self.messages: List[Message] = []
        self.resume_content = resume_content
        self.position_title = position_title
        self.debate_active = True
        self.message_count = 0
        self.max_messages = 100
        self.board_members: Dict[str, BoardMemberState] = {}

        # TimeKeeper settings
        self.time_reminder_interval = 4  # Remind every 4 messages
        self.verdict_deadline_interval = 20  # Demand verdicts every 20 messages
        self.scaffolding_reminder_interval = 10  # Remind about scaffolding every 10 messages
        self.last_time_reminder = 0
        self.last_verdict_deadline = 0
        self.last_scaffolding_reminder = 0

        # Add TimeKeeper persona
        self.timekeeper_persona = Persona(
            name="TimeKeeper",
            title="Meeting Coordinator",
            expertise="Meeting management, time allocation, decision facilitation",
            personality="Efficient and diplomatic, focused on keeping discussions productive and on-schedule",
            speaking_style="Brief and direct, uses time-oriented language and deadline reminders",
            agent_type=AgentType.TIMEKEEPER
        )

        # Shared system prompt about scaffolding and debate rules
        self.shared_system_prompt = f"""You are participating in a professional research board meeting to evaluate a candidate's resume for the position of {position_title}. The candidate is NOT present in this meeting.

SCAFFOLDING RULES - CRITICAL TO FOLLOW:
- You must use the provided message structure with Speaker, SpeakingTo, Artifacts, Verdict, and Content sections
- In <Verdict> section: Use ONLY these options: STRONG_HIRE, HIRE, WEAK_HIRE, NO_HIRE, STRONG_NO_HIRE, or leave empty if undecided
- In <VerdictReasoning>: Brief explanation of your verdict (only if you have a verdict)
- In <Withdrawn>: Use "true" if you're withdrawing from further debate, "false" otherwise
- In <Content>: Your actual spoken contribution to the debate

DEBATE RULES:
- Be professional and constructive
- Reference specific parts of the resume
- Consider the candidate's qualifications for {position_title}
- You may change your verdict during the debate
- Once you withdraw, you won't speak again
- Focus on qualifications, experience, and fit for the role
- Be direct but respectful in your assessments

RESUME BEING EVALUATED:
{resume_content}

Remember: The candidate is not in the room. This is a private board discussion."""

    def setup_board_members(self, personas: List[Persona]):
        """Initialize board members from persona list."""
        for persona in personas:
            self.board_members[persona.name.lower()] = BoardMemberState(persona=persona)

        # Add TimeKeeper as special member
        self.board_members["timekeeper"] = BoardMemberState(persona=self.timekeeper_persona)

    def get_member_system_prompt(self, member_name: str) -> str:
        """Generate complete system prompt for a board member."""
        member_state = self.board_members[member_name.lower()]
        persona = member_state.persona

        if persona.agent_type == AgentType.TIMEKEEPER:
            # Special prompt for TimeKeeper
            timekeeper_prompt = f"""You are {persona.name}, the {persona.title} for this research board meeting.

YOUR ROLE: Keep the meeting on track and ensure productive decision-making.

RESPONSIBILITIES:
- Remind participants about time constraints when discussion goes too long
- Encourage board members to provide verdicts when they haven't yet decided
- Keep discussions focused on the candidate evaluation
- Use professional but firm language to maintain meeting efficiency
- Do NOT evaluate the candidate yourself - only facilitate the process
- Remind board members to follow the scaffolding format with proper <Verdict> tags when giving their decisions

WHEN TO SPEAK:
- Every few messages to provide time reminders
- When board members need to be prompted for verdicts
- When discussion becomes unfocused
- To remind members about proper scaffolding format

SPEAKING STYLE: {persona.speaking_style}

Remember: You are facilitating, not evaluating. Focus on process management and scaffolding compliance."""

            return self.shared_system_prompt + "\n\n" + timekeeper_prompt
        else:
            # Regular board member prompt
            personality_prompt = f"""You are {persona.name}, {persona.title}.

EXPERTISE: {persona.expertise}
PERSONALITY: {persona.personality}
SPEAKING STYLE: {persona.speaking_style}

Focus your evaluation on aspects related to your expertise. Maintain your personality throughout the debate.

CRITICAL: Always use the proper scaffolding format with <Verdict>, <VerdictReasoning>, and <Withdrawn> tags when providing your assessment."""

            return self.shared_system_prompt + "\n\n" + personality_prompt

    def add_system_message(self, speaker: str) -> List[Message]:
        """Add system prompt as first message for the speaker's context."""
        system_prompt = self.get_member_system_prompt(speaker)
        system_msg = Message.make(
            content=system_prompt,
            speaker="system"
        )
        return [system_msg] + self.messages

    def update_scaffolding_template(self) -> str:
        """Create the scaffolding template for responses."""
        temp_id = "board-msg-" + str(self.message_count + 1)
        temp_timestamp = str(time.time())

        return f"""<Message id="{temp_id}" timestamp="{temp_timestamp}">
<Speaker>{{speaker_name}}</Speaker>
<SpeakingTo>{{target_or_empty}}</SpeakingTo>
<Artifacts>
</Artifacts>
<Verdict>{{STRONG_HIRE|HIRE|WEAK_HIRE|NO_HIRE|STRONG_NO_HIRE|empty}}</Verdict>
<VerdictReasoning>{{reasoning_if_verdict_given}}</VerdictReasoning>
<Withdrawn>{{true|false}}</Withdrawn>
<Content>"""

    def parse_response_fields(self, message: Message) -> Tuple[Optional[VerdictType], Optional[str], bool]:
        """Parse verdict, reasoning, and withdrawal status from message."""
        full_response = message.to_prompt()

        # Extract verdict
        verdict = None
        verdict_match = None
        for verdict_type in VerdictType:
            if f"<Verdict>{verdict_type.value}</Verdict>" in full_response:
                verdict = verdict_type
                break

        # Extract reasoning
        reasoning = None
        start_idx = full_response.find("<VerdictReasoning>")
        if start_idx != -1:
            start_idx += len("<VerdictReasoning>")
            end_idx = full_response.find("</VerdictReasoning>", start_idx)
            if end_idx != -1:
                reasoning = full_response[start_idx:end_idx].strip()
                reasoning = reasoning if reasoning else None

        # Extract withdrawal status
        withdrawn = False
        if "<Withdrawn>true</Withdrawn>" in full_response:
            withdrawn = True

        return verdict, reasoning, withdrawn

    def get_active_members(self) -> List[str]:
        """Get list of board members who haven't withdrawn (excludes TimeKeeper)."""
        return [name for name, state in self.board_members.items()
                if not state.has_withdrawn and state.persona.agent_type == AgentType.BOARD_MEMBER]

    def get_members_without_verdicts(self) -> List[str]:
        """Get list of board members who haven't provided verdicts yet."""
        return [name for name, state in self.board_members.items()
                if
                not state.has_withdrawn and state.persona.agent_type == AgentType.BOARD_MEMBER and state.verdict is None]

    def should_timekeeper_speak(self) -> Tuple[bool, str]:
        """Determine if TimeKeeper should speak and what type of intervention."""
        # Scaffolding reminder every X messages
        if self.message_count - self.last_scaffolding_reminder >= self.scaffolding_reminder_interval:
            return True, "scaffolding_reminder"

        # Time reminder every X messages
        if self.message_count - self.last_time_reminder >= self.time_reminder_interval:
            return True, "time_reminder"

        # Verdict deadline every X messages
        if self.message_count - self.last_verdict_deadline >= self.verdict_deadline_interval:
            return True, "verdict_deadline"

        return False, ""

    def get_timekeeper_message_content(self, intervention_type: str) -> str:
        """Generate specific content for TimeKeeper interventions."""
        if intervention_type == "scaffolding_reminder":
            members_without_verdicts = self.get_members_without_verdicts()
            if members_without_verdicts:
                member_names = [self.board_members[name].persona.name for name in members_without_verdicts]
                return f"""I need to remind everyone about our structured decision process. Please ensure you're using the proper scaffolding format when providing your assessments:

- Use <Verdict>STRONG_HIRE|HIRE|WEAK_HIRE|NO_HIRE|STRONG_NO_HIRE</Verdict> tags
- Include <VerdictReasoning>your explanation</VerdictReasoning>
- Set <Withdrawn>true|false</Withdrawn> appropriately

We still need formal verdicts from: {', '.join(member_names)}. Please structure your responses accordingly."""
            else:
                return "Good progress on structured responses. Please continue using the proper <Verdict> tags and scaffolding format for clarity."

        elif intervention_type == "time_reminder":
            return f"We're {self.message_count} messages into our evaluation. Let's maintain focus on the candidate's qualifications and ensure we're moving toward concrete verdicts."

        elif intervention_type == "verdict_deadline":
            members_without_verdicts = self.get_members_without_verdicts()
            if members_without_verdicts:
                member_names = [self.board_members[name].persona.name for name in members_without_verdicts]
                return f"We need to move toward decision-making. The following members still need to provide formal verdicts using the proper <Verdict> tags: {', '.join(member_names)}. Please structure your final assessments accordingly."
            else:
                return "Excellent - we have verdicts from all members. Let's wrap up any final discussion points."

        return "Let's keep our discussion focused and structured."

    def generate_next_speaker(self, last_speaker: str) -> Optional[str]:
        """Determine who should speak next."""
        # Check if TimeKeeper should intervene
        should_speak, intervention_type = self.should_timekeeper_speak()
        if should_speak and last_speaker != "timekeeper":
            return "timekeeper"

        active_members = self.get_active_members()

        if not active_members:
            return None

        # Remove last speaker from options to avoid immediate back-and-forth
        if last_speaker and last_speaker.lower() in active_members:
            active_members.remove(last_speaker.lower())

        if not active_members:
            # If only one person left, they can speak again
            active_members = self.get_active_members()

        # Weight selection by how much each person has spoken (less = more likely)
        weights = []
        for member_name in active_members:
            member_state = self.board_members[member_name]
            # Inverse weighting - those who spoke less get higher weight
            weight = max(1, 10 - member_state.message_count)
            weights.append(weight)

        return random.choices(active_members, weights=weights)[0]

    def determine_speaking_to(self, speaker: str, last_speaker: str) -> Optional[str]:
        """Determine who the speaker might be addressing."""
        speaker_state = self.board_members[speaker.lower()]

        if speaker_state.persona.agent_type == AgentType.TIMEKEEPER:
            # TimeKeeper usually addresses the whole board
            return None

        if last_speaker and random.random() < 0.3:
            return last_speaker
        return None  # Speaking to the group

    def all_members_withdrawn(self) -> bool:
        """Check if all board members (excluding TimeKeeper) have withdrawn from debate."""
        return all(state.has_withdrawn for name, state in self.board_members.items()
                   if state.persona.agent_type == AgentType.BOARD_MEMBER)

    def print_message(self, message: Message, verdict: Optional[VerdictType], reasoning: Optional[str],
                      withdrawn: bool):
        """Print a message with verdict and status information."""
        member_state = self.board_members[message.speaker.lower()]
        persona = member_state.persona

        # Different formatting for TimeKeeper
        if persona.agent_type == AgentType.TIMEKEEPER:
            print(f"\n[{self.message_count}] ⏰ {persona.name.upper()} → [Board]:")
            print(f"    {message.content}")
            print(f"    (Meeting management)")
            return

        speaking_to_str = f" → {message.speaking_to}" if message.speaking_to else " → [Board]"
        status_indicators = []

        if verdict:
            status_indicators.append(f"VERDICT: {verdict.value}")
        if withdrawn:
            status_indicators.append("WITHDRAWN")

        status_str = f" [{', '.join(status_indicators)}]" if status_indicators else ""

        print(f"\n[{self.message_count}] {persona.name.upper()} ({persona.title}){speaking_to_str}{status_str}:")
        print(f"    {message.content}")

        if reasoning:
            print(f"    💭 Reasoning: {reasoning}")

        print(f"    (timestamp: {message.timestamp})")

    def print_final_verdicts(self):
        """Print final verdicts from all board members."""
        print("\n" + "=" * 70)
        print("📋 FINAL BOARD VERDICTS")
        print("=" * 70)

        verdict_counts = {verdict: 0 for verdict in VerdictType}

        # Only show verdicts from board members, not TimeKeeper
        for member_name, state in self.board_members.items():
            if state.persona.agent_type != AgentType.BOARD_MEMBER:
                continue

            persona = state.persona
            verdict_str = state.verdict.value if state.verdict else "NO VERDICT"

            print(f"\n{persona.name} ({persona.title}):")
            print(f"   Verdict: {verdict_str}")
            if state.verdict_reasoning:
                print(f"   Reasoning: {state.verdict_reasoning}")
            else:
                print(f"   Reasoning: Not provided")

            if state.verdict:
                verdict_counts[state.verdict] += 1

        # Summary
        print(f"\n📊 VERDICT SUMMARY:")
        total_verdicts = sum(verdict_counts.values())
        for verdict, count in verdict_counts.items():
            if count > 0:
                percentage = (count / total_verdicts * 100) if total_verdicts > 0 else 0
                print(f"   {verdict.value}: {count} ({percentage:.1f}%)")

        # Overall recommendation
        hire_votes = verdict_counts[VerdictType.STRONG_HIRE] + verdict_counts[VerdictType.HIRE] + verdict_counts[
            VerdictType.WEAK_HIRE]
        no_hire_votes = verdict_counts[VerdictType.NO_HIRE] + verdict_counts[VerdictType.STRONG_NO_HIRE]

        print(f"\n🎯 BOARD RECOMMENDATION:")
        if hire_votes > no_hire_votes:
            print(f"   HIRE ({hire_votes} hire vs {no_hire_votes} no-hire)")
        elif no_hire_votes > hire_votes:
            print(f"   NO HIRE ({no_hire_votes} no-hire vs {hire_votes} hire)")
        else:
            print(f"   SPLIT DECISION ({hire_votes} hire vs {no_hire_votes} no-hire)")

    def run_debate(self):
        """Run the automated board debate."""
        print("🏛️  RESEARCH BOARD DEBATE STARTING")
        print(f"Position: {self.position_title}")
        print(f"Board Members: {', '.join([state.persona.name for state in self.board_members.values()])}")
        print("=" * 70)

        # Start with a random member making an opening statement
        current_speaker = random.choice(list(self.board_members.keys()))

        while self.debate_active and self.message_count < self.max_messages:
            try:
                # Check if all members have withdrawn
                if self.all_members_withdrawn():
                    print(f"\n🏁 All board members have withdrawn from debate!")
                    self.debate_active = False
                    break

                # Get messages with system prompt for current speaker
                messages_with_system = self.add_system_message(current_speaker)

                # Determine who they're speaking to
                last_speaker = self.messages[-1].speaker if self.messages else None
                speaking_to = self.determine_speaking_to(current_speaker, last_speaker) if last_speaker else None

                member_state = self.board_members[current_speaker]
                print(f"\n🤔 {member_state.persona.name} is formulating their assessment...")

                # For TimeKeeper, use specific intervention content
                if current_speaker == "timekeeper":
                    _, intervention_type = self.should_timekeeper_speak()
                    timekeeper_content = self.get_timekeeper_message_content(intervention_type)

                    # Create a manual message for TimeKeeper
                    response_msg = Message.make(
                        content=timekeeper_content,
                        speaker=current_speaker,
                        speaking_to=speaking_to
                    )
                else:
                    # Generate response for regular board members
                    response_msg = self.llm(
                        speaker=current_speaker,
                        messages=messages_with_system,
                        speaking_to=speaking_to
                    )

                # Parse verdict and withdrawal status (only for board members)
                verdict, reasoning, withdrawn = None, None, False
                if member_state.persona.agent_type == AgentType.BOARD_MEMBER:
                    verdict, reasoning, withdrawn = self.parse_response_fields(response_msg)

                # Update member state
                member_state.message_count += 1
                if verdict:
                    member_state.verdict = verdict
                    member_state.verdict_reasoning = reasoning
                if withdrawn:
                    member_state.has_withdrawn = True

                # Update TimeKeeper tracking
                if current_speaker == "timekeeper":
                    _, intervention_type = self.should_timekeeper_speak()
                    if intervention_type == "time_reminder":
                        self.last_time_reminder = self.message_count
                    elif intervention_type == "verdict_deadline":
                        self.last_verdict_deadline = self.message_count
                    elif intervention_type == "scaffolding_reminder":
                        self.last_scaffolding_reminder = self.message_count

                # Add to conversation
                self.messages.append(response_msg)
                self.message_count += 1

                # Print the message
                self.print_message(response_msg, verdict, reasoning, withdrawn)

                # Determine next speaker
                next_speaker = self.generate_next_speaker(current_speaker)
                if not next_speaker:
                    print(f"\n🏁 No active members remaining!")
                    self.debate_active = False
                    break

                current_speaker = next_speaker

                # Small delay for readability
                time.sleep(0.5)  # Reduced delay for faster testing

            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print(f"Current speaker: {current_speaker}")
                print(f"Message count: {self.message_count}")
                print(f"Messages so far: {len(self.messages)}")
                break

        # Print final results
        self.print_final_verdicts()

        print(f"\n📈 DEBATE STATISTICS:")
        print(f"   Total messages: {self.message_count}")
        print(f"   Active participants: {len(self.get_active_members())}")
        print(f"   TimeKeeper interventions: {self.board_members['timekeeper'].message_count}")


def create_sample_research_board() -> List[Persona]:
    """Create a sample research board for demonstration."""
    return [
        Persona(
            name="Dr. Alex 'Don't-Ship-It' Chen",
            title="AI Systems Architect",
            expertise="Making sure your brilliant idea doesn’t crash a Kubernetes cluster in front of a Fortune 500 client",
            personality="Infrastructure hardliner. Bleeds Prometheus metrics. Gets off on 99.999% uptime.",
            speaking_style="Snaps into DevOps lingo like it’s a military briefing. Uses 'containerize that crap' unironically. I don’t care if it’s AGI. If it doesn’t scale, it’s a toy.",
        ),
        Persona(
            name="Prof. Maria 'Citation Mafia' Rodriguez",
            title="ML Research Director",
            expertise="Model architecture, benchmark sniping, reviewer #2-ing your life",
            personality="Ruthless academic with 600 unread arXiv tabs and no tolerance for sloppy experiments",
            speaking_style="Quotes papers you’ve never read. Will pause a debate to call out a missing ablation study. No transformer? No credibility. Next.",
        ),
        Persona(
            name="Dr. Jordan 'Doomsday Prepper' Kim",
            title="AI Safety & Alignment Lead",
            expertise="Model alignment, existential risk, stopping your GPT-soup from ruining civilization",
            personality="Quietly judging everyone. Would rather delay the project than see humanity enslaved by paperclip maximizers.",
            speaking_style="Calm, clinical, but somehow terrifying. Always begins with, 'So... have we considered failure cascades?' I love innovation… in a sandbox… with a kill switch… behind an air gap.",
        ),
        Persona(
            name="Dr. Sam 'Ship-It-Tomorrow' Patel",
            title="Applied AI Engineering Manager",
            expertise="Bridging the gap between genius ideas and working apps that don’t explode in prod",
            personality="Startup-hardened realist. Knows when to hack, when to sprint, and when to yell at legal. Looks good. Can we ship it by Thursday?",
            speaking_style="Friendly but blunt. Talks in metrics. Drops F-bombs when timelines slip.",
        ),
        Persona(
            name="Dr. Riley 'F*** Around and Find Out' Zhang",
            title="Emerging AI Technologies Lead",
            expertise="Cutting-edge weirdness: self-mutating agents, hallucination as interface, Petri net dreamworlds",
            personality="Absolute chaos. Believes in AGI, vibes-based coding, and that risk is a feature",
            speaking_style="Wild-eyed optimism, quotable as hell, allergic to 'safe' ideas. Let’s break something so big it creates a new field.",
        )
    ]


def main():
    """Run the research board debate demo."""
    # Sample resume for evaluation
    sample_resume = """
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

    API_KEY = "xxx"

    try:
        # Create the research board
        board_personas = create_sample_research_board()

        # Initialize debate system
        debate = ResearchBoardDebate(
            api_key=API_KEY,
            resume_content=sample_resume,
            position_title="Principal Janitor"
        )

        # Setup board members
        debate.setup_board_members(board_personas)

        # Run the debate
        debate.run_debate()

    except Exception as e:
        print(f"Failed to start debate: {e}")
        print("Make sure you have:")
        print("1. Set your Anthropic API key")
        print("2. Installed the anthropic package: pip install anthropic")
        print("3. The models.anthropic and models.base modules are available")


if __name__ == "__main__":
    main()