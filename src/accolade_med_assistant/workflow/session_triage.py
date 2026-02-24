from dataclasses import dataclass

from accolade_med_assistant.intake.patient_intake_mock import MockPatientIntakeAgent
from accolade_med_assistant.models.types import Intake, PatientSessionInput, TriageResult
from accolade_med_assistant.triage.triage_router import TriageRouter


@dataclass
class SessionTriageOutput:
    intake: Intake
    triage: TriageResult


class SessionTriageWorkflow:
    def __init__(self, intake_agent: MockPatientIntakeAgent | None = None, router: TriageRouter | None = None) -> None:
        self.intake_agent = intake_agent or MockPatientIntakeAgent()
        self.router = router or TriageRouter()

    def run(self, session: PatientSessionInput) -> SessionTriageOutput:
        intake = self.intake_agent.build_intake(session)
        triage = self.router.run(intake)
        return SessionTriageOutput(intake=intake, triage=triage)
