from typing import List, TypedDict
import warnings

# LangGraph currently pulls langchain-core, which emits a Python 3.14 warning
# for legacy pydantic v1 compatibility internals. Suppress this specific warning.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from langgraph.graph import END, START, StateGraph

from accolade_med_assistant.intake.patient_intake_mock import MockPatientIntakeAgent
from accolade_med_assistant.models.types import Intake, PatientSessionInput


class IntakeState(TypedDict, total=False):
    session: PatientSessionInput
    transcript: List[str]
    aggregated_text: str
    symptoms: List[str]
    duration: str | None
    missing_fields: List[str]
    next_question: str
    next_field: str
    turn_count: int
    complete: bool
    camera_done: bool
    camera_scene_description: str | None
    abrasion_findings: List[str]
    intake: Intake


class IterativeIntakeWorkflow:
    def __init__(self, intake_agent: MockPatientIntakeAgent | None = None, max_turns: int = 4) -> None:
        self.intake_agent = intake_agent or MockPatientIntakeAgent()
        self.max_turns = max_turns
        self.graph = self._build_graph()

    def run(self, session: PatientSessionInput) -> Intake:
        final_state = self.graph.invoke({"session": session})
        return final_state["intake"]

    def _build_graph(self):
        graph = StateGraph(IntakeState)
        graph.add_node("initialize", self._initialize_node)
        graph.add_node("collect_signals", self._collect_signals_node)
        graph.add_node("assess_completeness", self._assess_completeness_node)
        graph.add_node("ask_followup", self._ask_followup_node)
        graph.add_node("camera_tool", self._camera_tool_node)
        graph.add_node("finalize_intake", self._finalize_node)

        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "collect_signals")
        graph.add_edge("collect_signals", "assess_completeness")
        graph.add_conditional_edges(
            "assess_completeness",
            self._route_after_assessment,
            {
                "camera_tool": "camera_tool",
                "ask_followup": "ask_followup",
                "finalize": "finalize_intake",
            },
        )
        graph.add_edge("camera_tool", "collect_signals")
        graph.add_edge("ask_followup", "collect_signals")
        graph.add_edge("finalize_intake", END)

        return graph.compile()

    def _initialize_node(self, state: IntakeState) -> IntakeState:
        session = state["session"]
        transcript = [f"Patient opening statement: {session.patient_message or 'No initial statement.'}"]
        return {
            "transcript": transcript,
            "aggregated_text": session.patient_message or "",
            "duration": session.duration,
            "turn_count": 0,
            "camera_done": False,
            "camera_scene_description": None,
            "abrasion_findings": [],
        }

    def _collect_signals_node(self, state: IntakeState) -> IntakeState:
        transcript = state.get("transcript", [])
        aggregated_text = state.get("aggregated_text", "")
        symptoms = self.intake_agent.extract_symptoms_from_text(aggregated_text, transcript)
        return {"symptoms": symptoms}

    def _assess_completeness_node(self, state: IntakeState) -> IntakeState:
        duration = state.get("duration")
        symptoms = state.get("symptoms", [])
        session = state["session"]

        missing: List[str] = []
        if not symptoms or symptoms == ["general malaise"]:
            missing.append("symptoms")
        if not duration:
            missing.append("duration")

        turn_count = state.get("turn_count", 0)
        complete = not missing or turn_count >= self.max_turns

        question = ""
        field = ""
        if not complete and missing:
            field = missing[0]
            question = self._question_for_field(field)

        return {
            "missing_fields": missing,
            "complete": complete,
            "next_question": question,
            "next_field": field,
        }

    def _route_after_assessment(self, state: IntakeState) -> str:
        session = state["session"]
        camera_done = state.get("camera_done", False)
        if session.camera_enabled and not camera_done:
            return "camera_tool"

        if state.get("complete", False):
            return "finalize"

        return "ask_followup"

    def _camera_tool_node(self, state: IntakeState) -> IntakeState:
        session = state["session"]
        text = state.get("aggregated_text", session.patient_message)
        camera_description = self.intake_agent.camera_tool_describe(session, text)
        abrasions = self.intake_agent.camera_tool_abrasion_scan(session, text)
        transcript = list(state.get("transcript", []))
        transcript.append(
            f"Tool:camera observation => {camera_description}; abrasions => {abrasions or ['none']}"
        )
        return {
            "camera_done": True,
            "camera_scene_description": camera_description,
            "abrasion_findings": abrasions,
            "transcript": transcript,
        }

    def _ask_followup_node(self, state: IntakeState) -> IntakeState:
        session = state["session"]
        question = state.get("next_question", "Can you provide more details?")
        field = state.get("next_field", "general")
        answer = session.mock_followup_answers.get(field, "unknown")

        transcript = list(state.get("transcript", []))
        transcript.append(f"Q: {question} A: {answer}")

        aggregated = state.get("aggregated_text", "")
        aggregated = f"{aggregated} {answer}".strip()

        update: IntakeState = {
            "transcript": transcript,
            "aggregated_text": aggregated,
            "turn_count": state.get("turn_count", 0) + 1,
        }
        if field == "duration" and answer != "unknown":
            update["duration"] = answer
        return update

    def _finalize_node(self, state: IntakeState) -> IntakeState:
        session = state["session"]
        intake = Intake(
            age_years=session.age_years,
            sex=session.sex,
            symptoms=state.get("symptoms", ["general malaise"]),
            first_name=session.first_name,
            last_name=session.last_name,
            id_number=session.id_number,
            duration=state.get("duration"),
            notes=state.get("aggregated_text", "").strip(),
            interview_transcript=state.get("transcript", []),
            camera_scene_description=state.get("camera_scene_description"),
            abrasion_findings=state.get("abrasion_findings", []),
            scan_findings=session.scan_findings,
            scan_image_path=session.scan_image_path,
            environment=session.environment,
        )
        return {"intake": intake}

    def _question_for_field(self, field: str) -> str:
        if field == "duration":
            return "How long have these symptoms been present?"
        if field == "symptoms":
            return "Can you describe your main symptoms in a few words?"
        return "Can you provide more details?"
