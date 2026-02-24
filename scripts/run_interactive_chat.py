from pathlib import Path
import argparse
import os
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accolade_med_assistant.models.types import ConversationSessionInput
from accolade_med_assistant.config_runtime import load_runtime_settings
from accolade_med_assistant.workflow.conversation_agent_graph import ConversationAgentGraph


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSession ended.")
        return "exit"


def ask_optional_int(prompt: str):
    raw = ask(prompt)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print("Invalid number, leaving empty.")
        return None


def yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = ask(f"{prompt} {suffix} ").lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes"}


def collect_symptoms() -> str:
    print("\nAssistant: Please describe your main symptom.")
    primary = ask("You: ")
    if primary.lower() in {"exit", "quit"}:
        return primary

    symptoms = [primary] if primary else []
    while True:
        extra = ask("Assistant: Any other symptom? (enter to continue): ")
        if not extra:
            break
        symptoms.append(extra)

    if not symptoms:
        return ""
    if len(symptoms) == 1:
        return symptoms[0]
    return "Symptoms: " + "; ".join(symptoms)


def run_followup_loop(assistant: ConversationAgentGraph, output) -> None:
    print("\nAssistant: Ask follow-up questions if you want more detail (enter to finish).")
    while True:
        q = ask("You (follow-up): ")
        if not q:
            break
        response = assistant.answer_followup_question(q, output)
        print(response)


def configure_clean_demo_output(enabled: bool) -> None:
    if not enabled:
        return

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    warnings.filterwarnings("ignore")
    warnings.filterwarnings(
        "ignore",
        message="resource_tracker: There appear to be .* leaked semaphore objects.*",
        category=UserWarning,
    )

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="WERKA interactive chat")
    parser.add_argument(
        "--show-model-logs",
        action="store_true",
        help="Show model loading progress bars and warnings.",
    )
    args = parser.parse_args()
    configure_clean_demo_output(enabled=not args.show_model_logs)

    print("WERKA Interactive Chat")
    print("Type 'exit' at symptom prompt to stop.\n")

    assistant = ConversationAgentGraph()
    runtime = load_runtime_settings()

    first_name = ask("First name: ") or None
    last_name = ask("Last name: ") or None
    id_number = ask("ID number: ") or None
    age_years = ask_optional_int("Age: ")
    sex = ask("Sex: ") or None

    env = runtime.environment
    print(f"Using configured location: {runtime.location_label}")
    print(f"Using configured environment: {env}")
    camera_enabled = yes_no("Enable camera checks and abrasion scan?", True)

    camera_estimated_age = None
    camera_estimated_sex = None
    if camera_enabled and yes_no("Provide mock camera estimate for identity cross-check?", False):
        camera_estimated_age = ask_optional_int("Camera estimated age: ")
        camera_estimated_sex = ask("Camera estimated sex: ") or None

    use_medgemma = yes_no("Use MedGemma optional interpretation?", False)

    while True:
        patient_message = collect_symptoms()
        if patient_message.lower() in {"exit", "quit"}:
            print("Session ended.")
            break

        duration = ask("Assistant: How long have these symptoms been present? (optional): ") or None
        image_path = ask("Assistant: Optional scan image path/URL (enter to skip): ") or None

        verification_followup_answers: dict[str, str] = {}
        if camera_enabled and (camera_estimated_age is not None or camera_estimated_sex):
            verification_followup_answers["identity_check"] = (
                ask("Assistant: Camera mismatch may exist. Please confirm identity details: ")
                or "No clarification provided."
            )

        session = ConversationSessionInput(
            first_name=first_name,
            last_name=last_name,
            id_number=id_number,
            age_years=age_years,
            sex=sex,
            environment=env,
            patient_message=patient_message,
            duration=duration,
            camera_enabled=camera_enabled,
            camera_estimated_age=camera_estimated_age,
            camera_estimated_sex=camera_estimated_sex,
            verification_followup_answers=verification_followup_answers,
            additional_questions=(),
            scan_image_path=image_path,
            use_medgemma=use_medgemma,
            mock_followup_answers={"duration": duration or "unknown"},
        )

        output = assistant.run(session)

        print("\nAssistant: Verification notes")
        for note in output.verification_notes:
            print(f"- {note}")

        print("\nAssistant: Recommendation summary")
        print(f"- Urgency: {output.triage.urgency}")
        print(f"- Next step: {output.triage.recommended_next_step}")
        if output.triage.guardrail_triggered:
            print("- Guardrail override: yes")
            for reason in output.triage.guardrail_reasons:
                print(f"  * {reason}")

        print("\nAssistant: Conversation")
        for line in output.dialogue:
            print(line)
        run_followup_loop(assistant, output)

        if not yes_no("\nStart another case?", False):
            print("Session finished.")
            break


if __name__ == "__main__":
    main()
