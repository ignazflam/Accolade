from pathlib import Path
from pprint import pprint
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accolade_med_assistant.models.types import ConversationSessionInput
from accolade_med_assistant.workflow.conversation_agent_graph import ConversationAgentGraph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conversation flow demo")
    parser.add_argument("--verbose", action="store_true", help="Print full objects")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast local settings (CPU + shorter generation + skip MedGemma image call).",
    )
    args = parser.parse_args()

    if args.fast:
        os.environ.setdefault("ACCOLADE_LLM_DEVICE", "cpu")
        os.environ.setdefault("ACCOLADE_DECIDER_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        os.environ.setdefault("ACCOLADE_LLM_MAX_NEW_TOKENS", "96")
        os.environ.setdefault("ACCOLADE_LLM_MAX_TIME_SECONDS", "20")
        os.environ.setdefault("ACCOLADE_DISABLE_MEDGEMMA", "1")

    session = ConversationSessionInput(
        first_name="Anna",
        last_name="Kowalska",
        id_number="PL-998877",
        age_years=50,
        sex="male",
        camera_estimated_age=24,
        camera_estimated_sex="female",
        verification_followup_answers={"identity_check": "I confirmed my ID and age details with the nurse."},
        environment="remote_village",
        patient_message="I have shortness of breath and cough for 3 days.",
        duration="2 days",
        scan_image_path=None
        if args.fast
        else "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        use_medgemma=False,
        additional_questions=(
            "Can you repeat what I should do now?",
            "Where should I go for transport support?",
        ),
    )

    assistant = ConversationAgentGraph()
    output = assistant.run(session)

    if args.verbose:
        print("=== Verification Notes ===")
        pprint(output.verification_notes)
        print("\n=== Intake ===")
        pprint(output.intake)
        print("\n=== Triage ===")
        pprint(output.triage)
    else:
        print("=== Demo Summary ===")
        print(f"Urgency: {output.triage.urgency}")
        print(f"Recommended next step: {output.triage.recommended_next_step}")
        print("Immediate actions:")
        for idx, action in enumerate(output.triage.immediate_actions[:4], start=1):
            print(f"{idx}. {action}")
        print(f"Guardrail triggered: {output.triage.guardrail_triggered}")

    print("\n=== Model Routing ===")
    print(f"MedGemma called: {output.medgemma_called}")
    print(f"Decision reason: {output.medgemma_decision_reason}")
    if args.verbose:
        print(f"MedGemma feedback: {output.medgemma_feedback}")
    print("\n=== Dialogue ===")
    for line in output.dialogue:
        print(line)


if __name__ == "__main__":
    main()
