from pathlib import Path
from pprint import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accolade_med_assistant.workflow.clinical_assistant_graph import ClinicalAssistantGraph


def main() -> None:
    assistant = ClinicalAssistantGraph()

    output = assistant.run_from_symptoms(
        "I have shortness of breath, cough and small skin abrasion with bleeding.",
        age_years=52,
        sex="female",
        environment="remote_village",
        duration="2 days",
        image_path_or_url="https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        first_name="Anna",
        last_name="Kowalska",
        id_number="PL-998877",
    )

    print("=== Intake JSON ===")
    pprint(output.intake)
    print("\n=== Recommendations ===")
    pprint(output.triage)


if __name__ == "__main__":
    main()
