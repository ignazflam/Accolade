from dataclasses import asdict
from typing import List, TypedDict
import warnings

# LangGraph currently pulls langchain-core, which emits a Python 3.14 warning
# for legacy pydantic v1 compatibility internals. Suppress this specific
# upstream warning so app output stays clean on 3.14.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from langgraph.graph import END, START, StateGraph

from accolade_med_assistant.config import TriageConfig, default_environment_profiles
from accolade_med_assistant.data.patient_repository import LocalPatientRepository, PatientRecord
from accolade_med_assistant.inference.local_llm import LocalLLMClient
from accolade_med_assistant.models.types import Intake, TriageResult


class TriageState(TypedDict, total=False):
    intake: Intake
    guardrail_triggered: bool
    guardrail_reasons: List[str]
    patient_verified: bool
    patient_record_found: bool
    patient_history: List[str]
    relevant_scans: List[str]
    combined_text: str
    urgency: str
    rationale: List[str]
    immediate_actions: List[str]
    recommended_next_step: str
    environment_guidance: List[str]
    llm_prompt: str
    assistant_summary: str
    result: TriageResult


class TriageRouter:
    def __init__(
        self,
        config: TriageConfig | None = None,
        llm: LocalLLMClient | None = None,
        patient_repo: LocalPatientRepository | None = None,
    ) -> None:
        self.config = config or TriageConfig()
        self.llm = llm or LocalLLMClient()
        self.patient_repo = patient_repo or LocalPatientRepository()
        self.environment_profiles = default_environment_profiles()
        self.graph = self._build_graph()

    def run(self, intake: Intake) -> TriageResult:
        final_state = self.graph.invoke({"intake": intake})
        return final_state["result"]

    def _build_graph(self):
        graph = StateGraph(TriageState)
        graph.add_node("verify_patient", self._verify_patient_node)
        graph.add_node("fetch_patient_context", self._fetch_patient_context_node)
        graph.add_node("prepare_context", self._prepare_context_node)
        graph.add_node("guardrail_screen", self._guardrail_screen_node)
        graph.add_node("assign_priority", self._assign_priority_node)
        graph.add_node("build_recommendations", self._build_recommendations_node)
        graph.add_node("apply_environment_constraints", self._apply_environment_constraints_node)
        graph.add_node("summarize", self._summarize_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "verify_patient")
        graph.add_edge("verify_patient", "fetch_patient_context")
        graph.add_edge("fetch_patient_context", "prepare_context")
        graph.add_edge("prepare_context", "guardrail_screen")
        graph.add_conditional_edges(
            "guardrail_screen",
            self._route_after_guardrail,
            {
                "guardrail_triggered": "build_recommendations",
                "standard_priority": "assign_priority",
            },
        )
        graph.add_edge("assign_priority", "build_recommendations")
        graph.add_edge("build_recommendations", "apply_environment_constraints")
        graph.add_edge("apply_environment_constraints", "summarize")
        graph.add_edge("summarize", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _route_after_guardrail(self, state: TriageState) -> str:
        if state.get("guardrail_triggered", False):
            return "guardrail_triggered"
        return "standard_priority"

    def _verify_patient_node(self, state: TriageState) -> TriageState:
        intake = state["intake"]
        verified = bool(intake.first_name and intake.last_name and intake.id_number)
        return {"patient_verified": verified}

    def _fetch_patient_context_node(self, state: TriageState) -> TriageState:
        intake = state["intake"]
        if not state.get("patient_verified", False):
            return {
                "patient_record_found": False,
                "patient_history": [],
                "relevant_scans": [],
            }

        record = self.patient_repo.find_patient(
            intake.first_name or "",
            intake.last_name or "",
            intake.id_number or "",
        )
        if record is None:
            return {
                "patient_record_found": False,
                "patient_history": [],
                "relevant_scans": [],
            }

        return {
            "patient_record_found": True,
            "patient_history": list(record.history),
            "relevant_scans": self._select_relevant_scans(record, intake),
        }

    def _prepare_context_node(self, state: TriageState) -> TriageState:
        intake = state["intake"]
        patient_history = state.get("patient_history", [])
        relevant_scans = state.get("relevant_scans", [])
        rationale: List[str] = []
        if state.get("patient_verified", False):
            if state.get("patient_record_found", False):
                rationale.append("Patient identity verified and local history was retrieved.")
            else:
                rationale.append("Patient identity provided, but no local record was found.")
        else:
            rationale.append("Patient identity was not fully provided; proceeding without history lookup.")

        return {
            "combined_text": self._combined_text(intake, patient_history, relevant_scans),
            "rationale": rationale,
        }

    def _guardrail_screen_node(self, state: TriageState) -> TriageState:
        intake = state["intake"]
        combined_text = state["combined_text"]
        rationale = list(state.get("rationale", []))
        reasons: List[str] = []

        hard_guardrail_terms = (
            "unconscious",
            "seizure",
            "stroke",
            "one-sided weakness",
            "suicidal",
            "coughing blood",
            "heavy bleeding",
            "possible active bleeding signal detected",
            "severe chest pain",
            "shortness of breath",
            "difficulty breathing",
        )

        for term in hard_guardrail_terms:
            if term in combined_text:
                reasons.append(f"Detected guardrail symptom: {term}.")

        spo2 = intake.vitals.spo2_percent
        if spo2 is not None and spo2 < self.config.emergency_spo2_threshold:
            reasons.append(
                f"Detected critical SpO2: {spo2}% below emergency threshold {self.config.emergency_spo2_threshold}%."
            )

        if reasons:
            rationale.append("Safety guardrail override applied.")
            return {
                "guardrail_triggered": True,
                "guardrail_reasons": reasons,
                "urgency": "emergency",
                "rationale": rationale,
            }

        return {
            "guardrail_triggered": False,
            "guardrail_reasons": [],
            "rationale": rationale,
        }

    def _assign_priority_node(self, state: TriageState) -> TriageState:
        intake = state["intake"]
        combined_text = state["combined_text"]
        rationale = list(state.get("rationale", []))

        urgency = "routine"
        if self._is_emergency(combined_text, intake):
            urgency = "emergency"
            rationale.append("Emergency symptom or critical oxygen level detected.")
        elif self._is_urgent(combined_text, intake):
            urgency = "urgent"
            rationale.append("Potentially serious symptom pattern detected.")
        else:
            rationale.append("No hard emergency triggers detected from provided inputs.")

        if intake.scan_findings:
            rationale.append("Scan findings were included and considered for prioritization.")

        return {
            "urgency": urgency,
            "rationale": rationale,
        }

    def _build_recommendations_node(self, state: TriageState) -> TriageState:
        urgency = state["urgency"]
        return {
            "immediate_actions": self._actions_for_urgency(urgency),
            "recommended_next_step": self._next_step_for_urgency(urgency),
        }

    def _apply_environment_constraints_node(self, state: TriageState) -> TriageState:
        intake = state["intake"]
        urgency = state["urgency"]
        actions = list(state["immediate_actions"])
        next_step = state["recommended_next_step"]
        rationale = list(state["rationale"])

        profile = self.environment_profiles.get(intake.environment, self.environment_profiles["standard"])
        env_guidance: List[str] = [profile.guidance_note]

        if profile.care_constraints:
            env_guidance.extend(profile.care_constraints)

        if intake.environment == "remote_village":
            if urgency == "emergency":
                next_step = "Immediate nurse-led stabilization and transport coordination"
                actions.append("Nurse should monitor airway, breathing, circulation continuously until transfer.")
            elif urgency == "urgent":
                next_step = "Nurse assessment today with earliest feasible referral plan"
                actions.append("Define referral trigger list if travel cannot happen immediately.")
            else:
                actions.append("Schedule nurse re-check within 24 hours due to limited physician access.")
            rationale.append("Recommendations adapted for remote village constraints.")

        elif intake.environment == "limited_access_poland":
            if urgency == "emergency":
                next_step = "Immediate SOR/ER referral with available local transport"
            elif urgency == "urgent":
                next_step = "Same-day NPL/SOR assessment and stepwise low-cost diagnostics"
            else:
                next_step = "Primary care follow-up with staged diagnostics based on affordability"

            actions.append("Prefer available lower-cost diagnostics first when clinically safe.")
            actions.append("If MRI is unavailable locally, refer to nearest city only when it changes management.")
            rationale.append("Recommendations adapted for limited service and budget constraints.")

        else:
            actions.append(profile.escalation_note)

        return {
            "immediate_actions": actions,
            "recommended_next_step": next_step,
            "rationale": rationale,
            "environment_guidance": env_guidance,
        }

    def _summarize_node(self, state: TriageState) -> TriageState:
        intake = state["intake"]
        urgency = state["urgency"]
        rationale = state["rationale"]
        actions = state["immediate_actions"]
        next_step = state["recommended_next_step"]
        environment_guidance = state.get("environment_guidance", [])
        patient_history = state.get("patient_history", [])
        relevant_scans = state.get("relevant_scans", [])
        guardrail_reasons = state.get("guardrail_reasons", [])

        llm_prompt = self._build_llm_prompt(
            intake,
            urgency,
            rationale,
            actions,
            next_step,
            environment_guidance,
            patient_history,
            relevant_scans,
            guardrail_reasons,
        )
        llm_summary = self.llm.summarize(llm_prompt, scan_image_path=intake.scan_image_path)
        summary = llm_summary or self._default_summary(urgency, actions, next_step)

        return {
            "llm_prompt": llm_prompt,
            "assistant_summary": summary,
        }

    def _finalize_node(self, state: TriageState) -> TriageState:
        return {
            "result": TriageResult(
                urgency=state["urgency"],
                guardrail_triggered=state.get("guardrail_triggered", False),
                guardrail_reasons=state.get("guardrail_reasons", []),
                patient_verified=state.get("patient_verified", False),
                patient_record_found=state.get("patient_record_found", False),
                patient_history=state.get("patient_history", []),
                relevant_scans=state.get("relevant_scans", []),
                rationale=state["rationale"],
                immediate_actions=state["immediate_actions"],
                recommended_next_step=state["recommended_next_step"],
                assistant_summary=state["assistant_summary"],
            )
        }

    def _combined_text(
        self, intake: Intake, patient_history: List[str] | None = None, relevant_scans: List[str] | None = None
    ) -> str:
        symptoms_text = " ".join(intake.symptoms or [])
        interview_text = " ".join(intake.interview_transcript or [])
        camera_text = intake.camera_scene_description or ""
        abrasion_text = " ".join(intake.abrasion_findings or [])
        history_text = " ".join(patient_history or [])
        scans_text = " ".join(relevant_scans or [])
        return (
            f"{symptoms_text} {intake.notes} {intake.scan_findings or ''} "
            f"{interview_text} {camera_text} {abrasion_text} {history_text} {scans_text}"
        ).lower()

    def _select_relevant_scans(self, record: PatientRecord, intake: Intake) -> List[str]:
        if not record.scans:
            return []

        scan_entries = list(record.scans)
        symptom_words = {
            token
            for symptom in intake.symptoms
            for token in symptom.lower().split()
            if len(token) > 3
        }
        if not symptom_words:
            return scan_entries[-2:]

        matched = [scan for scan in scan_entries if any(word in scan.lower() for word in symptom_words)]
        if matched:
            return matched[:3]
        return scan_entries[-2:]

    def _is_emergency(self, text: str, intake: Intake) -> bool:
        if any(keyword in text for keyword in self.config.emergency_keywords):
            return True
        spo2 = intake.vitals.spo2_percent
        if spo2 is not None and spo2 < self.config.emergency_spo2_threshold:
            return True
        return False

    def _is_urgent(self, text: str, intake: Intake) -> bool:
        if any(keyword in text for keyword in self.config.urgent_keywords):
            return True
        spo2 = intake.vitals.spo2_percent
        if spo2 is not None and spo2 < self.config.urgent_spo2_threshold:
            return True
        return False

    def _actions_for_urgency(self, urgency: str) -> List[str]:
        if urgency == "emergency":
            return [
                "Seek emergency care immediately.",
                "Do not delay for home treatment.",
                "If available, keep monitoring breathing and consciousness.",
            ]
        if urgency == "urgent":
            return [
                "Arrange in-person clinical review as soon as possible (same day).",
                "Track symptoms and vitals every 2-4 hours.",
                "Escalate to emergency if symptoms worsen.",
            ]
        return [
            "Supportive care and close monitoring at home.",
            "Hydration, rest, and symptom log.",
            "Escalate if new red-flag symptoms appear.",
        ]

    def _next_step_for_urgency(self, urgency: str) -> str:
        if urgency == "emergency":
            return "Immediate emergency transfer"
        if urgency == "urgent":
            return "Same-day clinical evaluation"
        return "Routine follow-up in 24-72 hours"

    def _build_llm_prompt(
        self,
        intake: Intake,
        urgency: str,
        rationale: List[str],
        actions: List[str],
        next_step: str,
        environment_guidance: List[str],
        patient_history: List[str],
        relevant_scans: List[str],
        guardrail_reasons: List[str],
    ) -> str:
        history = patient_history if patient_history else ["No prior history found in local DB."]
        scans = relevant_scans if relevant_scans else ["No relevant prior scans found in local DB."]
        return (
            "You are a medical triage assistant. Summarize this case for a field worker in plain language. "
            "Do not provide definitive diagnosis.\n"
            f"Intake: {asdict(intake)}\n"
            f"Urgency: {urgency}\n"
            f"Rationale: {rationale}\n"
            f"Immediate actions: {actions}\n"
            f"Next step: {next_step}\n"
            f"Environment guidance: {environment_guidance}\n"
            f"Patient history context: {history}\n"
            f"Relevant prior scans context: {scans}\n"
            f"Guardrail reasons: {guardrail_reasons}\n"
        )

    def _default_summary(self, urgency: str, actions: List[str], next_step: str) -> str:
        return (
            f"Preliminary triage level: {urgency}. "
            f"Immediate actions: {' '.join(actions)} "
            f"Recommended next step: {next_step}."
        )
