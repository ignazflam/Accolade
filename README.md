# Accolade Medical Assistant (POC)

Offline-first triage assistant proof of concept.

## Structure

- `src/accolade_med_assistant/triage/triage_router.py`: LangGraph-based routing flow
- `src/accolade_med_assistant/models/types.py`: intake/result dataclasses
- `scripts/run_demo.py`: example run
- `tests/test_triage_router.py`: basic rule behavior tests

## LangGraph flow

The router is composed of explicit nodes:

1. `verify_patient`
2. `fetch_patient_context`
3. `prepare_context`
4. `guardrail_screen` (hard clinical emergency override)
5. `assign_priority` (runs only if guardrail does not trigger)
6. `build_recommendations`
7. `apply_environment_constraints`
8. `summarize`
9. `finalize`

## Local patient DB

Patient identity verification and history/scans lookup use:

- `/Users/krzysztof/Documents/Accolade/data/input/patients.json`

Required intake fields for verification:

- `first_name`
- `last_name`
- `id_number`

## Environment-aware profiles

Set `Intake.environment` to one of:

- `standard`: normal hospital/referral pathways
- `remote_village`: deep rural setting with nurse-led care and limited physician/imaging access
- `limited_access_region`: constrained diagnostics/budget access, with stepwise feasible referrals

## MedGemma backend

`LocalLLMClient` now supports local MedGemma summarization in the `summarize` node.

Enable it before running:

```bash
export ACCOLADE_LLM_BACKEND=medgemma
export ACCOLADE_LLM_DEVICE=mps
export ACCOLADE_LLM_DTYPE=float32
```

Then pass an image path or URL in `Intake.scan_image_path` for multimodal summarization.

## Run

```bash
cd /Users/krzysztof/Documents/Accolade
source .venv/bin/activate
python scripts/run_demo.py
```

Run patient-facing mock intake (LLM interview + camera description + abrasion scan):

```bash
cd /Users/krzysztof/Documents/Accolade
source .venv/bin/activate
python scripts/run_patient_mock_demo.py
```

The intake stage is iterative: it collects data, checks completeness, asks follow-up questions when needed, can call a mock camera tool, and only then emits the final intake JSON for triage.

Run unified single-node assistant graph (describe symptoms -> recommendations, optional image):

```bash
cd /Users/krzysztof/Documents/Accolade
source .venv/bin/activate
python scripts/run_assistant_node_demo.py
```

Run conversation-first flow (iterative verification -> intake -> optional MedGemma -> control follow-up):

```bash
cd /Users/krzysztof/Documents/Accolade
source .venv/bin/activate
python scripts/run_conversation_flow_demo.py
```

Run interactive terminal mode (manual data entry + assistant prompts):

```bash
cd /Users/krzysztof/Documents/Accolade
source .venv/bin/activate
python scripts/run_interactive_chat.py
```

Location/environment are loaded from config (not prompted each run):

- `/Users/krzysztof/Documents/Accolade/data/input/app_config.json`

## Test

```bash
cd /Users/krzysztof/Documents/Accolade
source .venv/bin/activate
export PYTHONPATH=/Users/krzysztof/Documents/Accolade/src
python -m unittest -v tests/test_triage_router.py
```

If MedGemma is disabled or fails to load, the app falls back to an offline-safe static summary.
