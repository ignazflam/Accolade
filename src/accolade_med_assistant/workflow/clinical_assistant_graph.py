from dataclasses import dataclass
from typing import TypedDict
import warnings

# LangGraph currently pulls langchain-core, which emits a Python 3.14 warning
# for legacy pydantic v1 compatibility internals. Suppress this specific warning.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from langgraph.graph import END, START, StateGraph

from accolade_med_assistant.intake.iterative_intake_workflow import IterativeIntakeWorkflow
from accolade_med_assistant.models.types import Intake, PatientSessionInput, TriageResult
from accolade_med_assistant.triage.triage_router import TriageRouter


@dataclass
class AssistantOutput:
    intake: Intake
    triage: TriageResult


class AssistantState(TypedDict, total=False):
    session: PatientSessionInput
    output: AssistantOutput


class ClinicalAssistantGraph:
    """Single-node runnable graph: symptom description -> recommendations."""

    def __init__(
        self,
        intake_workflow: IterativeIntakeWorkflow | None = None,
        triage_router: TriageRouter | None = None,
    ) -> None:
        self.intake_workflow = intake_workflow or IterativeIntakeWorkflow()
        self.triage_router = triage_router or TriageRouter()
        self.graph = self._build_graph()

    def run(self, session: PatientSessionInput) -> AssistantOutput:
        final_state = self.graph.invoke({"session": session})
        return final_state["output"]

    def run_from_symptoms(
        self,
        symptoms_text: str,
        *,
        age_years: int | None,
        sex: str | None,
        environment: str = "standard",
        duration: str | None = None,
        image_path_or_url: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        id_number: str | None = None,
    ) -> AssistantOutput:
        session = PatientSessionInput(
            first_name=first_name,
            last_name=last_name,
            id_number=id_number,
            age_years=age_years,
            sex=sex,
            environment=environment,
            patient_message=symptoms_text,
            duration=duration,
            scan_image_path=image_path_or_url,
        )
        return self.run(session)

    def _build_graph(self):
        graph = StateGraph(AssistantState)
        graph.add_node("intake_and_triage", self._intake_and_triage_node)
        graph.add_edge(START, "intake_and_triage")
        graph.add_edge("intake_and_triage", END)
        return graph.compile()

    def _intake_and_triage_node(self, state: AssistantState) -> AssistantState:
        session = state["session"]
        intake = self.intake_workflow.run(session)
        triage = self.triage_router.run(intake)
        return {"output": AssistantOutput(intake=intake, triage=triage)}
