# 🔗 Chain of Debate Framework


**[WIP - Evolving Architecture]** A powerful, general-purpose multi-agent orchestration system that enables AI experts to engage in structured debates through sequential goal progression, with advanced features like whisper communications and automated moderation.

## 🎯 What This Is

This isn't just another chatbot system - it's a **flexible debate engine** that can handle any complex evaluation or decision-making scenario. Whether you're evaluating job candidates, research proposals, investment opportunities, policy decisions, medical diagnoses, or creative projects, this framework provides the infrastructure for multi-expert analysis.

**The core insight:** Complex decisions benefit from **multiple expert perspectives** examining the problem from different angles. The system orchestrates structured conversations where each AI agent maintains their own expertise, personality, and memory while participating in group decision-making processes.

## 🔗 Why Chain of Debate? (The Technical Problem)

**The Challenge:** Multi-agent AI conversations are notoriously unstable. Agents go off-topic, ignore instructions, fail to reach conclusions, or get stuck in loops. Most systems either devolve into chaos or require heavy human intervention to stay on track.

**The Core Issue:** Traditional multi-agent systems lack structured goal progression and state management. Without clear objectives and automatic transitions, conversations drift aimlessly or collapse when agents can't maintain coherent long-term behavior.

**Our Approach:** Sequential goal chains with enforced scaffolding create conversation stability through:
- **Structured Transitions**: Goals process one at a time with automatic progression
- **State Reset Mechanisms**: Agent participation status resets between goals while preserving decision history  
- **Scaffolded Communication**: XML-like message structure prevents format drift and enables reliable parsing
- **Authority Hierarchy**: TimeKeeper agent with escalating intervention prevents endless discussions
- **Validation Pipeline**: Real-time message checking catches and corrects deviations

**The Result:** Stable, multi-goal conversations that consistently reach structured conclusions. Agents maintain personality and expertise across extended interactions while the system guarantees progression toward defined objectives.

**Why This Matters:** Reliable multi-agent orchestration enables complex AI collaboration for evaluation, decision-making, and analysis scenarios that previously required human moderation to remain coherent.

### Why Chain of Debates?

- **Sequential Processing:** Goals are tackled one at a time, allowing deep focus on each aspect
- **Structured Transitions:** Automatic progression from goal to goal with state management
- **Machine-Readable Outcomes:** Verdicts and reasoning captured in structured format
- **Systematic Testing:** Run identical scenarios through different expert configurations

## 🏗️ Architecture Evolution

### Current Framework Features

- **🔗 Sequential Goal Chains** - Process debate objectives one at a time with automatic transitions
- **🎭 Rich Persona System** - AI agents with distinct expertise, personalities, and speaking styles  
- **📋 Scaffolded Communication** - Enforced XML-like message structure with automatic validation
- **🤫 Whisper Mechanics** - Private side conversations between specific agents
- **⏱️ Smart TimeKeeper** - Escalating intervention system with configurable pressure levels
- **✅ Structured Verdicts** - Machine-readable decisions with reasoning requirements
- **👁️ Message Visibility** - Automatic filtering based on whisper permissions
- **🔍 Validation Pipeline** - Extensible message format and content checking

### Recent Major Upgrades

- **Abstract Base Architecture** - Clean separation between orchestration logic and domain implementations
- **Enhanced Scaffolding System** - Automatic injection of format examples into LLM prompts
- **Goal State Management** - Proper handling of agent state across goal transitions
- **Authority Structure** - Clear TimeKeeper authority with mandatory compliance
- **Whisper Analytics** - Comprehensive tracking of private communication patterns

## 🚀 Quick Start

```python
from agents.debate_chain import (
    Persona, ChainOfDebate, create_resume_verdict_config, 
    Goal, DebateTimeKeeperConfig
)

# Define expert personas
personas = [
    Persona(
        name="Dr. Sarah Chen",
        title="Technical Lead",
        expertise="Software architecture and system design", 
        personality="Detail-oriented and thorough",
        speaking_style="Technical and precise"
    ),
    Persona(
        name="Marcus Truth",
        title="Engineering Manager",
        expertise="Team leadership and hiring",
        personality="Brutally honest, fact-focused", 
        speaking_style="Direct and blunt, cites evidence"
    )
]

# Define sequential goals
goals = [
    Goal("technical_assessment", "Evaluate technical qualifications and skills"),
    Goal("cultural_fit", "Assess team alignment and collaboration potential"),
    Goal("growth_potential", "Analyze learning ability and career trajectory")
]

# Configure the debate
debate = ChainOfDebate(
    api_key="your-anthropic-api-key",
    debate_topic="Senior Engineer Candidate Evaluation",
    context_content="[Your evaluation context here]",
    verdict_config=create_resume_verdict_config(),
    goals=goals,
    timekeeper_config=DebateTimeKeeperConfig(
        intervention_interval=4,
        force_verdict_threshold=25
    )
)

# Run the sequential debate
debate.setup_agents(personas)
results = debate.run_debate()

print(f"Goals completed: {len(results['completed_goals'])}")
print(f"Final verdicts: {results['verdicts']}")
```

## 🔧 Technical Deep Dive

### Message Scaffolding System

Every agent communication follows strict XML-like scaffolding:

```xml
<Message id="unique-id" timestamp="2024-01-15T10:30:00">
<Speaker>agent_name</Speaker>
<SpeakingTo>target_agent</SpeakingTo>  <!-- Optional -->
<Whisper>true</Whisper>               <!-- For private messages -->
<Artifacts></Artifacts>
<Content>The actual message content</Content>
</Message>
```

The following are how the agents provide verdicts

```xml
<Verdict>APPROVE</Verdict>             <!-- Domain-specific -->
<VerdictReasoning>Detailed reasoning</VerdictReasoning>
<Withdrawn>false</Withdrawn>
```

### Sequential Goal Processing

1. **Goal 1 Activation** - All agents engage in structured debate
2. **Verdict Collection** - TimeKeeper enforces decision deadlines
3. **Automatic Transition** - When all agents withdraw, next goal activates
4. **State Reset** - Agent participation status resets, verdict history preserved
5. **Repeat Process** - Continue until all goals completed

### Smart TimeKeeper System

The TimeKeeper operates with escalating pressure:

- **Messages 1-15**: Gentle format reminders and progress updates
- **Messages 15-25**: Insistent demands for verdicts from undecided agents  
- **Messages 25-35**: Urgent deadline warnings with direct targeting
- **Messages 35+**: Force verdict completion mode

### Whisper Communication

Private channels enable strategic interactions:

```python
# Agents can whisper confidential information
response = agent.generate_whisper(
    target="Dr. Chen",
    content="I have concerns about this candidate's background check"
)
# Only sender and target see whisper messages
```

## 🎛️ Configuration & Customization

### Built-in Verdict Types

```python
# Resume/Hiring Evaluation
create_resume_verdict_config()
# Options: EXCELLENT_FIT, GOOD_FIT, ADEQUATE, POOR_FIT, REJECT

# Research Paper Review
create_research_verdict_config() 
# Options: BREAKTHROUGH, SIGNIFICANT, INCREMENTAL, INSUFFICIENT, FLAWED

# Business Proposal Assessment
create_proposal_verdict_config()
# Options: APPROVE, APPROVE_WITH_CONDITIONS, REVISE_AND_RESUBMIT, DECLINE
```

### Custom Validation Rules

```python
class CustomValidityChecker(ValidityChecker):
    def check(self, message: Message, config: Any, participant_state: Any) -> RejectionResult:
        # Your domain-specific validation
        if not self.meets_custom_criteria(message):
            return RejectionResult.invalid("Custom validation failed")
        return RejectionResult.valid()
```

### Extending the Framework

Create new debate types by extending the abstract base:

```python
class MedicalConsultation(AgentOrchestrator):
    def get_validation_config(self):
        return MedicalValidationConfig()
    
    def parse_custom_fields(self, response: str):
        # Extract medical-specific fields
        return {
            'diagnosis': extract_diagnosis(response),
            'confidence': extract_confidence(response),
            'recommended_tests': extract_tests(response)
        }
```

## 🧪 Testing & Validation

### Built-in Test Scenarios

```bash
# Test goal transition mechanics
python tests/goal_transition_test.py

# Test whisper communication patterns
python tests/whisper_test.py

# Test with challenging/adversarial personas  
python tests/controversial_evaluation_test.py
```

### Systematic Analysis

The framework enables powerful systematic testing:

- **Run identical cases** through different expert compositions
- **Analyze personality impact** on decision outcomes
- **Track whisper networks** and private influence patterns
- **Measure consensus formation** across different scenarios

## 🎯 Use Cases & Applications

### Evaluation Scenarios
- **Hiring Decisions** - Multi-stakeholder candidate assessment
- **Academic Review** - Peer review with diverse expert panels
- **Investment Analysis** - Due diligence across multiple dimensions
- **Medical Diagnosis** - Specialist consultation and second opinions

### Research Applications  
- **Group Decision Dynamics** - Study how expert composition affects outcomes
- **Consensus Formation** - Analyze how agreements emerge in groups
- **Information Flow** - Track how whispers and private channels influence decisions
- **Personality Impact** - Measure how individual traits affect group dynamics

### Business Applications
- **Strategic Planning** - Multi-department perspective integration
- **Risk Assessment** - Cross-functional team evaluation
- **Product Development** - Stakeholder alignment processes
- **Policy Formation** - Multi-expert policy analysis

## 🛠️ Development Status

### Current Status: WIP - Active Development

**Stable Components:**
- ✅ Core orchestration engine
- ✅ Scaffolded communication system
- ✅ Sequential goal processing
- ✅ Whisper mechanics
- ✅ Verdict tracking and validation

**In Progress:**
- 🔧 Advanced persona generation
- 🔧 Enhanced analytics dashboard
- 🔧 Performance optimization
- 🔧 Extended domain templates

**Planned Features:**
- 🔮 Dynamic expert generation from role descriptions  
- 🔮 Conversation summarization and compression
- 🔮 Multi-modal artifact support
- 🔮 Real-time collaboration interfaces
- 🔮 Integration with external knowledge bases
