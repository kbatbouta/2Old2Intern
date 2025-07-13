# 🔗2Old2Intern

**[WIP - Evolving Architecture]** A powerful, general-purpose multi-agent orchestration system that enables AI experts to engage in structured debates through sequential goal progression, with advanced features like whisper communications and automated moderation.

## 🎯 What This Is

This isn't just another chatbot system - it's a **flexible debate engine** that can handle any complex evaluation or decision-making scenario. Whether you're evaluating job candidates, research proposals, investment opportunities, policy decisions, medical diagnoses, or creative projects, this framework provides the infrastructure for multi-expert analysis.

**The core insight:** Complex decisions benefit from **multiple expert perspectives** examining the problem from different angles. The system orchestrates structured conversations where each AI agent maintains their own expertise, personality, and memory while participating in group decision-making processes.

## 🧠 Advanced Cognitive Features

### Three-Layer Communication System

The framework implements a sophisticated **three-tier information visibility model**:

1. **🌐 Public Messages** - Visible to all participants
2. **🤫 Whisper Messages** - Private communications between specific agents
3. **🧠 Private Thoughts** - Internal cognitive process visible only to the generating agent

### Private Thoughts Architecture

Each agent generates internal thoughts that simulate realistic cognitive processes:

```xml
<PrivateThoughts speaker="agent_name">
This candidate's background check shows some concerning gaps. I need to probe deeper 
into their employment history without revealing what I know publicly. Maybe I can 
coordinate with the security expert privately first to build a stronger case.
</PrivateThoughts>
```

**Key Features:**
- **Complete Privacy**: Only the generating agent can see their own thoughts
- **Strategic Planning**: Agents use thoughts to plan their approach and assess situations
- **Realistic Cognition**: Simulates the internal deliberation that precedes external communication
- **Coalition Strategy**: Agents think through alliance-building and information sharing tactics

### Enhanced Whisper Response System

The framework includes mechanisms to **increase whisper engagement**:

**Whisper Targeting Intelligence:**
- Agents analyze conversation context to identify strategic whisper opportunities
- Higher probability of whisper responses when sensitive information is shared
- Natural conversation threading where whispers often trigger follow-up whispers

**Response Probability Enhancement:**
```python
# Agents are more likely to respond to whispers that:
# - Share confidential information
# - Propose alliances or coordination
# - Request private consultation
# - Contain strategic intelligence
```

**Whisper Chain Formation:**
- Private conversations naturally develop into multi-message exchanges
- Agents remember previous whisper context when generating responses
- Strategic coordination often emerges through sustained private channels

## 🔧 Technical Deep Dive

### Message Scaffolding System

Every agent communication follows strict XML-like scaffolding:

```xml
<Message id="unique-id" timestamp="2024-01-15T10:30:00">
<Speaker>agent_name</Speaker>
<SpeakingTo>target_agent</SpeakingTo>  <!-- Optional -->
<Whisper>true</Whisper>               <!-- For private messages -->
<Artifacts></Artifacts>
<Content>
  <PrivateThoughts speaker="agent_name">
    Internal cognitive process and strategic planning
  </PrivateThoughts>
  
  The actual message content visible to intended recipients
</Content>
</Message>
```

**Verdict Structure:**
```xml
<Verdict>APPROVE</Verdict>             <!-- Domain-specific -->
<VerdictReasoning>Detailed reasoning</VerdictReasoning>
<Withdrawn>false</Withdrawn>
```

### Advanced Information Management

**Three-Tier Visibility System:**
1. **Public Layer**: Standard conversation visible to all participants
2. **Whisper Layer**: Private messages filtered by target recipient
3. **Cognitive Layer**: Internal thoughts visible only to the generating agent

**Message Filtering Pipeline:**
```python
def filter_messages_for_agent(self, messages: List[Message], target_agent: str) -> List[Message]:
    filtered = []
    for msg in messages:
        # Include public messages
        if not msg.is_whisper:
            filtered.append(msg)
        # Include whispers where agent is sender or target
        elif msg.speaker == target_agent or msg.speaking_to == target_agent:
            filtered.append(msg)
        # Include own private thoughts only
        elif self.contains_private_thoughts(msg) and msg.speaker == target_agent:
            filtered.append(msg)
    return filtered
```

### Whisper Engagement Mechanics

**Strategic Whisper Triggers:**
- Agents identify opportunities for private consultation
- Sensitive information naturally flows through whisper channels
- Coalition-building conversations emerge organically

**Response Amplification:**
- Whispers containing strategic keywords have higher response probability
- Private information sharing creates natural conversation threading
- Agents remember whisper context across multiple exchanges

**Alliance Formation Patterns:**
```python
# Common whisper patterns that trigger responses:
alliance_keywords = ['alliance', 'coordinate', 'strategy', 'together', 'confidential']
information_sharing = ['between you and me', 'privately', 'insider information']
strategic_planning = ['build a case', 'unified front', 'coordinate our approach']
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

### Enhanced Persona Configuration

```python
Persona(
    name="Marcus Gossip",
    title="HR Security Specialist",
    expertise="Background verification and workplace security",
    personality="Secretive and paranoid, loves sharing rumors through whispers",
    speaking_style="Frequently whispers sensitive information, creates alliances through private communications"
)
```

**Whisper-Prone Personality Traits:**
- Secretive information gatekeepers
- Strategic alliance builders  
- Paranoid security specialists
- Diplomatic coordinators
- Competitive analysts

### Advanced Analytics

**Whisper Pattern Analysis:**
```python
whisper_stats = {
    'total_whispers': count,
    'whisper_networks': agent_to_agent_mapping,
    'alliance_formations': detected_coalition_building,
    'information_cascades': private_info_flow_patterns
}
```

**Cognitive Process Tracking:**
- Internal thought pattern analysis
- Strategic planning identification
- Decision-making process visibility
- Coalition formation prediction

## 🧪 Testing & Validation

### Advanced Testing Scenarios

```bash
# Test three-tier communication system
python tests/communication_layers_test.py

# Test whisper engagement patterns
python tests/whisper_engagement_test.py

# Test private thoughts isolation
python tests/cognitive_privacy_test.py

# Test strategic alliance formation
python tests/coalition_building_test.py
```

**Whisper Behavior Validation:**
- Verify private thoughts remain isolated to generating agent
- Confirm whisper visibility rules are enforced
- Test whisper response amplification mechanisms
- Validate coalition formation detection

## 🛠️ Development Status

### Current Status: WIP - Active Development

**Stable Components:**
- ✅ Core orchestration engine
- ✅ Scaffolded communication system
- ✅ Sequential goal processing
- ✅ Three-tier visibility system
- ✅ Private thoughts architecture
- ✅ Enhanced whisper mechanics
- ✅ Verdict tracking and validation

**Recent Major Upgrades:**
- ✅ **Private Thoughts System** - Internal cognitive processes visible only to generating agent
- ✅ **Whisper Response Enhancement** - Improved probability of whisper engagement and threading
- ✅ **Coalition Detection** - Automatic identification of alliance formation patterns
- ✅ **Strategic Intelligence Tracking** - Analysis of information flow and influence patterns

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
