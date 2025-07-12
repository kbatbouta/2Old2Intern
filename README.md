IMPORTANT THIS IS STILL WIP

# [WIP] 🔧 AI Multi-Agent Debate System - Technical Overview

## **Core Architecture**

### **1. Dynamic Role-Based Message Routing 🎭**
The system uses a clever **speaker perspective switching** mechanism:
- When "Dr. Chen" is generating a response, all of HER previous messages become "assistant" role
- Everyone else's messages become "user" role  
- This gives each agent their own coherent conversation thread while sharing group context
- **Result:** Each AI maintains consistent personality and memory across the debate

### **2. Structured Message Scaffolding 📝**
Every message uses XML-like scaffolding that carries metadata:
```xml
<Message id="..." timestamp="...">
<Speaker>Dr. Sarah Chen</Speaker>
<SpeakingTo>Prof. Rodriguez</SpeakingTo>
<Artifacts></Artifacts>
<Verdict>HIRE</Verdict>
<VerdictReasoning>Strong publication record</VerdictReasoning>
<Withdrawn>false</Withdrawn>
<Content>I'm impressed by the h-index of 24...</Content>
</Message>
```

**Why this matters:**
- **Verdicts** are machine-readable, not buried in text
- **Withdrawal status** lets agents exit the debate
- **Speaking targets** create realistic conversational flow
- **Artifacts** can hold documents, charts, data

### **3. Intelligent Speaker Selection Logic 🎯**
```python
def generate_next_speaker():
    # TimeKeeper intervention check
    if message_count % 4 == 0: return "timekeeper"  
    
    # Weight by participation (less active = more likely to speak)
    weights = [max(1, 10 - member.message_count) for member in active_members]
    return weighted_random_choice(active_members, weights)
```

**Result:** Natural conversation flow where quiet members get pulled in

### **4. TimeKeeper Agent with State Tracking ⏰**
The TimeKeeper operates on **interval-based triggers**:
- **Time reminders:** Every 4 messages ("clock is ticking")
- **Verdict deadlines:** Every 20 messages ("need decisions now")
- **State awareness:** Knows who hasn't voted, who's withdrawn

**Implementation:**
```python
class AgentType(Enum):
    BOARD_MEMBER = "evaluates candidate"
    TIMEKEEPER = "manages process only"

# Different system prompts, different behaviors
if agent.type == TIMEKEEPER:
    prompt = "Facilitate meeting, don't evaluate candidate"
else:
    prompt = "Evaluate candidate using your expertise"
```

### **5. Dual System Prompt Architecture 🧠**
Each agent gets **two layers** of instructions:

**Shared Layer:** Scaffolding rules, debate format, resume content
```
- Use <Verdict> tags with specific options
- Reference specific resume sections  
- Professional meeting behavior
```

**Individual Layer:** Personality, expertise, speaking style
```
Dr. Chen: "Analytical, values rigorous methodology"
Prof. Rodriguez: "Strategic, focuses on leadership potential"  
```

### **6. Conversation State Management 📊**
```python
@dataclass
class BoardMemberState:
    verdict: Optional[VerdictType] = None
    verdict_reasoning: Optional[str] = None  
    has_withdrawn: bool = False
    message_count: int = 0
```

**Logic:** Debate continues until `all(member.has_withdrawn for member in board_members)`

## **The Technical Magic** ✨

### **Message Continuation vs. New Message Logic:**
- **Last message from same speaker:** Extract scaffolding, continue from `<Content>` tag
- **Last message from different speaker:** Create new scaffolding with fresh ID/timestamp
- **Empty conversation:** Auto-generate temp scaffolding to start

### **API Constraint Handling:**
- Anthropic requires alternating user/assistant messages
- System automatically inserts bridging messages when needed
- Maintains conversation context while satisfying API requirements

### **Verdict Extraction & State Updates:**
```python
def parse_response_fields(message):
    verdict = extract_between_tags(message, "Verdict")
    reasoning = extract_between_tags(message, "VerdictReasoning") 
    withdrawn = "<Withdrawn>true</Withdrawn>" in message
    return verdict, reasoning, withdrawn
```

## **Why This Architecture Works** 🎯

1. **Scalable:** Add new agent types by extending the enum and prompt templates
2. **Stateful:** Each agent remembers their positions and reasoning
3. **Structured:** Machine-readable decisions, not just conversational text  
4. **Realistic:** TimeKeeper adds meeting pressure and deadline management
5. **Flexible:** Same framework works for hiring, medical consultations, investment decisions

**Bottom line:** It's a **multi-agent orchestration system** where each AI maintains individual state and personality while participating in structured group decision-making with automatic process management.

## **TLDR - The Simple Logic**

Think of it like having five expert consultants in a conference room, but they're AI agents instead of real people. Each expert has their own specialty and personality - one focuses on technical skills, another on leadership, another on industry experience. When it's Dr. Chen's turn to speak, the system "becomes" her - it remembers everything she's said before and responds as her character would. Meanwhile, a meeting coordinator keeps track of time and makes sure everyone actually makes a decision instead of talking forever. The key insight is that each AI expert maintains their own perspective and memory throughout the entire discussion, just like real people would, but they can debate instantly without scheduling conflicts. The system automatically manages who speaks when, ensures decisions get made, and tracks all the votes and reasoning in a structured way that can be easily analyzed at the end.
