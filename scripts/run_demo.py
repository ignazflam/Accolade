from pathlib import Path
from pprint import pprint
import sys

# Allow running this script directly without manually exporting PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accolade_med_assistant.models.types import Intake, VitalSigns
from accolade_med_assistant.triage.triage_router import TriageRouter


def main() -> None:
    intake = Intake(
        first_name="Anna",
        last_name="Kowalska",
        id_number="PL-998877",
        age_years=52,
        sex="female",
        symptoms=["shortness of breath", "fever", "dry cough"],
        duration="2 days",
        notes="Breathlessness increasing this morning.",
        vitals=VitalSigns(temperature_c=38.2, heart_rate_bpm=112, spo2_percent=89),
        scan_findings="Bilateral patchy opacities described on chest X-ray.",
        scan_image_path="https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        environment="remote_village",
    )

    router = TriageRouter()
    result = router.run(intake)
    pprint(result)


if __name__ == "__main__":
    main()
