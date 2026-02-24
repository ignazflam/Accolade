from dataclasses import dataclass

from accolade_med_assistant.intake.iterative_intake_workflow import IterativeIntakeWorkflow
from accolade_med_assistant.intake.patient_intake_mock import MockPatientIntakeAgent
from accolade_med_assistant.models.types import Intake, PatientSessionInput, TriageResult
from accolade_med_assistant.triage.triage_router import TriageRouter
from accolade_med_assistant.workflow.clinical_assistant_graph import ClinicalAssistantGraph


@dataclass
class SessionTriageOutput:
    intake: Intake
    triage: TriageResult


class SessionTriageWorkflow:
    def __init__(self, intake_agent: MockPatientIntakeAgent | None = None, router: TriageRouter | None = None) -> None:
        intake_workflow = IterativeIntakeWorkflow(intake_agent=intake_agent or MockPatientIntakeAgent())
        triage_router = router or TriageRouter()
        self.assistant = ClinicalAssistantGraph(intake_workflow=intake_workflow, triage_router=triage_router)

    def run(self, session: PatientSessionInput) -> SessionTriageOutput:
        output = self.assistant.run(session)
        return SessionTriageOutput(intake=output.intake, triage=output.triage)
