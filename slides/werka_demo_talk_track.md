## WERKA 3-Minute Talk Track

### 0:00-0:20
- "WERKA is an offline-first medical assistant POC for constrained care settings."

### 0:20-0:45
- "It runs a LangGraph workflow: verification, intake, model routing, optional scan reasoning, and follow-up."
- "Qwen handles routing/follow-ups, and MedGemma is used for non-routine cases when enabled."

### 0:45-1:45
- Run:
  - `python /Users/krzysztof/Documents/Accolade/scripts/run_conversation_flow_demo.py`
- Narrate:
  - urgency
  - immediate actions
  - next step
  - model routing decision

### 1:45-2:25
- "The assistant handles follow-ups in context."
- Call out:
  - "Can you repeat what I should do now?"
  - Action recap response

### 2:25-3:00
- "WERKA combines local models with structured orchestration and environment-aware triage."
- "Next: stronger structured MedGemma outputs and production UI."
