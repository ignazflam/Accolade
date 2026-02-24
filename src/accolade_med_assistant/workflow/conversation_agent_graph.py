from dataclasses import replace
import re
from typing import List, TypedDict
import warnings
import os

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from langgraph.graph import END, START, StateGraph

from accolade_med_assistant.inference.dual_model_controller import DualModelController
from accolade_med_assistant.inference.local_llm import LocalLLMClient
from accolade_med_assistant.intake.iterative_intake_workflow import IterativeIntakeWorkflow
from accolade_med_assistant.models.types import ConversationRunOutput, ConversationSessionInput, Intake, TriageResult
from accolade_med_assistant.triage.triage_router import TriageRouter


class ConversationState(TypedDict, total=False):
    session: ConversationSessionInput
    verification_notes: List[str]
    verification_complete: bool
    verification_mismatch: bool
    intake: Intake
    triage: TriageResult
    medgemma_called: bool
    medgemma_decision_reason: str
    medgemma_feedback: str | None
    dialogue: List[str]
    question_index: int


class ConversationAgentGraph:
    """Conversation-first assistant flow.

    Flow:
    1) verification/assessment (iterative when mismatch)
    2) intake completion
    3) DeepSeek decides if MedGemma is needed
    4) optional MedGemma recommendations (can override default triage actions)
    5) control node response + follow-up loop
    6) finish
    """

    def __init__(
        self,
        intake_workflow: IterativeIntakeWorkflow | None = None,
        triage_router: TriageRouter | None = None,
        llm: LocalLLMClient | None = None,
    ) -> None:
        self.intake_workflow = intake_workflow or IterativeIntakeWorkflow()
        self.triage_router = triage_router or TriageRouter()
        self.llm = llm or LocalLLMClient()
        decider_model = os.getenv("ACCOLADE_DECIDER_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        self.model_controller = DualModelController(
            decider_llm=LocalLLMClient(
                model_name=decider_model,
                backend="deepseek",
                max_new_tokens=96,
            ),
            medgemma_llm=LocalLLMClient(model_name="google/medgemma-1.5-4b-it", backend="medgemma"),
        )
        self.graph = self._build_graph()

    def run(self, session: ConversationSessionInput) -> ConversationRunOutput:
        final_state = self.graph.invoke({"session": session})
        return ConversationRunOutput(
            intake=final_state["intake"],
            triage=final_state["triage"],
            verification_notes=final_state.get("verification_notes", []),
            medgemma_called=final_state.get("medgemma_called", False),
            medgemma_decision_reason=final_state.get("medgemma_decision_reason", ""),
            medgemma_feedback=final_state.get("medgemma_feedback"),
            dialogue=final_state.get("dialogue", []),
        )

    def answer_followup_question(self, question: str, output: ConversationRunOutput) -> str:
        return self._answer_followup_with_deepseek(
            question=question,
            triage=output.triage,
            medgemma_feedback=output.medgemma_feedback,
            verification_notes=list(output.verification_notes),
        )

    def _build_graph(self):
        graph = StateGraph(ConversationState)
        graph.add_node("verify_user", self._verify_user_node)
        graph.add_node("verification_followup", self._verification_followup_node)
        graph.add_node("generate_intake", self._generate_intake_node)
        graph.add_node("decide_medgemma", self._decide_medgemma_node)
        graph.add_node("medgemma_recommend", self._medgemma_recommend_node)
        graph.add_node("control_initial_response", self._control_initial_response_node)
        graph.add_node("control_followup", self._control_followup_node)

        graph.add_edge(START, "verify_user")
        graph.add_conditional_edges(
            "verify_user",
            self._route_after_verification,
            {
                "verification_followup": "verification_followup",
                "generate_intake": "generate_intake",
            },
        )
        graph.add_edge("verification_followup", "generate_intake")
        graph.add_edge("generate_intake", "decide_medgemma")
        graph.add_conditional_edges(
            "decide_medgemma",
            self._route_after_medgemma_decision,
            {
                "medgemma_recommend": "medgemma_recommend",
                "control_initial_response": "control_initial_response",
            },
        )
        graph.add_edge("medgemma_recommend", "control_initial_response")
        graph.add_conditional_edges(
            "control_initial_response",
            self._route_after_initial_response,
            {
                "control_followup": "control_followup",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "control_followup",
            self._route_followup_loop,
            {
                "control_followup": "control_followup",
                "end": END,
            },
        )

        return graph.compile()

    def _verify_user_node(self, state: ConversationState) -> ConversationState:
        session = state["session"]
        notes: List[str] = []
        mismatch = False

        if session.first_name and session.last_name and session.id_number:
            notes.append("Identity provided: name and ID present.")
        else:
            notes.append("Identity incomplete: proceeding with reduced verification confidence.")

        if session.camera_enabled and (session.camera_estimated_age is not None or session.camera_estimated_sex):
            if session.camera_estimated_age is not None and session.age_years is not None:
                if abs(session.camera_estimated_age - session.age_years) >= 15:
                    mismatch = True
                    notes.append(
                        f"Camera age estimate mismatch ({session.camera_estimated_age}) vs claimed age ({session.age_years})."
                    )
            if session.camera_estimated_sex and session.sex:
                if session.camera_estimated_sex.lower() != session.sex.lower():
                    mismatch = True
                    notes.append(
                        f"Camera sex estimate mismatch ({session.camera_estimated_sex}) vs claimed sex ({session.sex})."
                    )

        return {
            "verification_notes": notes,
            "verification_mismatch": mismatch,
            "verification_complete": not mismatch,
            "dialogue": [],
            "question_index": 0,
        }

    def _route_after_verification(self, state: ConversationState) -> str:
        if state.get("verification_mismatch", False):
            return "verification_followup"
        return "generate_intake"

    def _verification_followup_node(self, state: ConversationState) -> ConversationState:
        session = state["session"]
        notes = list(state.get("verification_notes", []))
        clarification = session.verification_followup_answers.get(
            "identity_check",
            "No clarification provided.",
        )
        notes.append(f"Verification follow-up answer: {clarification}")
        if clarification.strip() and clarification.strip().lower() != "no":
            notes.append("Verification accepted after follow-up.")
            complete = True
        else:
            notes.append("Verification unresolved; continuing with caution.")
            complete = False
        return {"verification_notes": notes, "verification_complete": complete}

    def _generate_intake_node(self, state: ConversationState) -> ConversationState:
        session = state["session"]
        intake = self.intake_workflow.run(session)
        triage = self.triage_router.run(intake)
        return {"intake": intake, "triage": triage}

    def _decide_medgemma_node(self, state: ConversationState) -> ConversationState:
        session = state["session"]
        intake = state["intake"]
        triage = state["triage"]
        disable_medgemma = os.getenv("ACCOLADE_DISABLE_MEDGEMMA", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if disable_medgemma:
            return {
                "medgemma_called": False,
                "medgemma_decision_reason": "Disabled by ACCOLADE_DISABLE_MEDGEMMA.",
            }

        if triage.urgency in {"urgent", "emergency"}:
            return {
                "medgemma_called": True,
                "medgemma_decision_reason": (
                    f"Automatic MedGemma call for non-routine triage ({triage.urgency})."
                ),
            }

        call_medgemma, reason = self.model_controller.should_call_medgemma(intake)
        if session.use_medgemma:
            call_medgemma = True
            reason = f"Forced by session flag. Original decision: {reason}"

        return {"medgemma_called": call_medgemma, "medgemma_decision_reason": reason}

    def _route_after_medgemma_decision(self, state: ConversationState) -> str:
        if state.get("medgemma_called", False):
            return "medgemma_recommend"
        return "control_initial_response"

    def _medgemma_recommend_node(self, state: ConversationState) -> ConversationState:
        intake = state["intake"]
        triage = state["triage"]

        parsed, raw = self.model_controller.medgemma_recommendations(intake)
        if parsed is None:
            if raw:
                if self._is_unusable_medgemma_text(raw):
                    return {
                        "triage": triage,
                        "medgemma_feedback": "MedGemma response was non-clinical/instructional; using baseline triage.",
                    }
                triage_overridden = self._override_triage_from_medgemma_text(triage, raw)
                return {"triage": triage_overridden, "medgemma_feedback": raw}
            return {"medgemma_feedback": raw}

        urgency = str(parsed.get("urgency", triage.urgency)).lower()
        if urgency not in {"emergency", "urgent", "routine"}:
            urgency = triage.urgency
        next_step = str(parsed.get("recommended_next_step", triage.recommended_next_step))

        actions = parsed.get("immediate_actions", triage.immediate_actions)
        if not isinstance(actions, list) or not actions:
            actions = list(triage.immediate_actions)
        actions = [str(a) for a in actions]

        rationale_text = str(parsed.get("rationale", "Recommendations generated by MedGemma."))

        triage_overridden = replace(
            triage,
            urgency=urgency,
            immediate_actions=actions,
            recommended_next_step=next_step,
            rationale=[*triage.rationale, f"MedGemma rationale: {rationale_text}"],
        )
        return {"triage": triage_overridden, "medgemma_feedback": raw}

    def _override_triage_from_medgemma_text(self, triage: TriageResult, raw: str) -> TriageResult:
        text = raw.strip()
        if not text:
            return triage

        lowered = text.lower()
        urgency = triage.urgency
        if "emergency" in lowered:
            urgency = "emergency"
        elif "urgent" in lowered:
            urgency = "urgent"
        elif "routine" in lowered:
            urgency = "routine"

        # Prefer explicit list-like lines that are actionable (avoid summary/analysis bullets).
        actions = []
        blocked_prefixes = (
            "patient:",
            "symptoms:",
            "interview:",
            "vitals:",
            "scan:",
            "environment:",
            "analysis:",
            "case summary:",
            "triage assessment:",
        )
        action_terms = (
            "go",
            "seek",
            "call",
            "monitor",
            "rest",
            "hydrate",
            "use",
            "take",
            "avoid",
            "start",
            "arrange",
            "transfer",
            "refer",
            "contact",
        )
        for line in text.splitlines():
            cleaned = line.strip()
            if re.match(r"^([-*]|\d+\.)\s+", cleaned):
                item = re.sub(r"^([-*]|\d+\.)\s+", "", cleaned).strip()
                lower_item = item.lower()
                if lower_item.startswith(blocked_prefixes):
                    continue
                if any(term in lower_item for term in action_terms):
                    actions.append(item)
        if not actions:
            sentences = [p.strip() for p in re.split(r"[.;]\s+", text) if p.strip()]
            for sentence in sentences:
                lower_sentence = sentence.lower()
                if lower_sentence.startswith(blocked_prefixes):
                    continue
                if any(term in lower_sentence for term in action_terms):
                    actions.append(sentence)
                if len(actions) >= 3:
                    break
        if not actions:
            actions = list(triage.immediate_actions)

        next_step = actions[0] if actions else triage.recommended_next_step
        return replace(
            triage,
            urgency=urgency,
            immediate_actions=actions,
            recommended_next_step=next_step,
            rationale=[*triage.rationale, f"MedGemma free-text rationale: {text[:300]}"],
        )

    def _control_initial_response_node(self, state: ConversationState) -> ConversationState:
        triage = state["triage"]
        medgemma_feedback = state.get("medgemma_feedback")
        notes = list(state.get("verification_notes", []))
        dialogue = list(state.get("dialogue", []))
        medgemma_called = state.get("medgemma_called", False)
        medgemma_decision_reason = state.get("medgemma_decision_reason", "")

        base_response = (
            f"Verification notes: {notes}. "
            f"MedGemma called: {medgemma_called} ({medgemma_decision_reason}). "
            f"Triage urgency: {triage.urgency}. "
            f"Recommended next step: {triage.recommended_next_step}."
        )
        if medgemma_feedback:
            base_response += f" MedGemma feedback: {medgemma_feedback}"

        dialogue.append(f"Assistant: {base_response}")
        return {"dialogue": dialogue}

    def _route_after_initial_response(self, state: ConversationState) -> str:
        session = state["session"]
        if session.additional_questions:
            return "control_followup"
        return "end"

    def _control_followup_node(self, state: ConversationState) -> ConversationState:
        session = state["session"]
        idx = state.get("question_index", 0)
        questions = list(session.additional_questions)
        dialogue = list(state.get("dialogue", []))
        triage = state["triage"]
        medgemma_feedback = state.get("medgemma_feedback")
        verification_notes = list(state.get("verification_notes", []))

        if idx >= len(questions):
            return {"question_index": idx, "dialogue": dialogue}

        question = questions[idx]
        dialogue.append(f"User: {question}")
        response = self._answer_followup_with_deepseek(
            question=question,
            triage=triage,
            medgemma_feedback=medgemma_feedback,
            verification_notes=verification_notes,
        )
        dialogue.append(response)
        return {"question_index": idx + 1, "dialogue": dialogue}

    def _route_followup_loop(self, state: ConversationState) -> str:
        session = state["session"]
        idx = state.get("question_index", 0)
        if idx < len(session.additional_questions):
            return "control_followup"
        return "end"

    def _is_general_question(self, question: str) -> bool:
        text = question.lower()
        medical_terms = (
            "pain",
            "fever",
            "breath",
            "bleeding",
            "cough",
            "stroke",
            "vomit",
            "symptom",
            "dose",
            "drug",
            "medicine",
        )
        return not any(term in text for term in medical_terms)

    def _answer_followup_with_deepseek(
        self,
        question: str,
        triage: TriageResult,
        medgemma_feedback: str | None,
        verification_notes: List[str],
    ) -> str:
        if self._is_action_recap_question(question):
            return f"Assistant: {self._format_action_recap(triage)}"

        medgemma_context = (medgemma_feedback or "").strip()
        if len(medgemma_context) > 1200:
            medgemma_context = medgemma_context[:1200] + "..."

        prompt = (
            "You are a concise clinical assistant handling patient follow-up questions.\n"
            "Answer directly in plain text (max 3 short sentences), no markdown, no bullet list, no chain-of-thought.\n"
            "If the user asks logistics or transport, provide practical local-next-step guidance.\n"
            "If the user asks medical follow-up, align with the triage urgency and recommended next step.\n"
            f"Verification notes: {verification_notes}\n"
            f"Triage urgency: {triage.urgency}\n"
            f"Triage recommended next step: {triage.recommended_next_step}\n"
            f"Triage immediate actions: {triage.immediate_actions}\n"
            f"MedGemma context (optional): {medgemma_context or 'none'}\n"
            f"User follow-up question: {question}\n"
            "Return only the assistant answer text."
        )

        answer = self.model_controller.decider_llm.generate_text(prompt)
        answer = (answer or "").strip()
        answer = re.sub(r"<unused\d+>\s*thought\b", "", answer, flags=re.IGNORECASE).strip()
        if self._looks_like_case_analysis(answer):
            answer = ""
        if answer.startswith("Preliminary support summary generated without a connected local LLM."):
            answer = ""
        if answer.lower().startswith("assistant:"):
            answer = answer.split(":", 1)[1].strip()

        if not answer:
            if self._is_action_recap_question(question):
                answer = self._format_action_recap(triage)
            elif self._is_general_question(question):
                answer = (
                    "This is a general support question. I can help with practical non-medical guidance, "
                    "logistics, and next steps."
                )
            else:
                answer = (
                    "Based on current triage, "
                    f"urgency remains {triage.urgency}. Next step: {triage.recommended_next_step}."
                )

        return f"Assistant: {answer}"

    def _is_action_recap_question(self, question: str) -> bool:
        text = question.lower()
        triggers = (
            "what should i do",
            "what do i do",
            "what should i do now",
            "what now",
            "repeat what i should do",
            "next step",
            "what are the steps",
            "what actions",
            "what should be done",
        )
        return any(t in text for t in triggers)

    def _format_action_recap(self, triage: TriageResult) -> str:
        actions = [a.strip() for a in triage.immediate_actions if str(a).strip()]
        top_actions = actions[:3]
        if top_actions:
            action_text = "; ".join(f"{i + 1}) {a}" for i, a in enumerate(top_actions))
            return (
                f"Urgency is {triage.urgency}. Do this now: {action_text}. "
                f"Next step: {triage.recommended_next_step}."
            )
        return f"Urgency is {triage.urgency}. Next step: {triage.recommended_next_step}."

    def _looks_like_case_analysis(self, answer: str) -> bool:
        lowered = answer.lower()
        markers = (
            "case analysis",
            "case summary",
            "triage assessment",
            "patient:",
            "symptoms:",
        )
        return any(m in lowered for m in markers)

    def _is_unusable_medgemma_text(self, text: str) -> bool:
        lowered = text.lower().strip()
        markers = (
            "the user wants me to act as",
            "output must be strict json",
            "return only",
            "do not include keys",
            "no markdown, no bullets",
            "provide a json output",
        )
        return any(m in lowered for m in markers)
