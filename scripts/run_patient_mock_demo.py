from pathlib import Path
from pprint import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accolade_med_assistant.models.types import PatientSessionInput
from accolade_med_assistant.workflow.session_triage import SessionTriageWorkflow


def main() -> None:
    session = PatientSessionInput(
        first_name="Anna",
        last_name="Kowalska",
        id_number="PL-998877",
        age_years=52,
        sex="female",
        environment="remote_village",
        duration="2 days",
        patient_message="I have cough, shortness of breath, and scraped my arm with minor bleeding.",
        camera_enabled=True,
    )

    workflow = SessionTriageWorkflow()
    output = workflow.run(session)

    print("=== Mock Intake (LLM + Camera) ===")
    pprint(output.intake)
    print("\n=== Triage Result ===")
    pprint(output.triage)


if __name__ == "__main__":
    main()
