import unittest
import os

from accolade_med_assistant.data.patient_repository import LocalPatientRepository
from accolade_med_assistant.inference.local_llm import LocalLLMClient
from accolade_med_assistant.models.types import Intake, PatientSessionInput, VitalSigns
from accolade_med_assistant.triage.triage_router import TriageRouter
from accolade_med_assistant.workflow.clinical_assistant_graph import ClinicalAssistantGraph
from accolade_med_assistant.workflow.conversation_agent_graph import ConversationAgentGraph
from accolade_med_assistant.workflow.session_triage import SessionTriageWorkflow


class TriageRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.getenv("ACCOLADE_LLM_BACKEND", "fallback").lower() == "medgemma":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("ACCOLADE_LLM_LOCAL_ONLY", "1")

        repo = LocalPatientRepository("/Users/krzysztof/Documents/Accolade/data/input/patients.json")
        llm = LocalLLMClient(
            backend=os.getenv("ACCOLADE_LLM_BACKEND", "fallback"),
            device=os.getenv("ACCOLADE_LLM_DEVICE", "mps"),
            dtype=os.getenv("ACCOLADE_LLM_DTYPE", "float32"),
            max_new_tokens=64,
        )
        # Load graph and (optionally) model once so sequential tests don't re-materialize weights.
        cls.router = TriageRouter(patient_repo=repo, llm=llm)

        if os.getenv("ACCOLADE_LLM_BACKEND", "fallback").lower() == "medgemma":
            warmup = Intake(age_years=30, sex="female", symptoms=["cough"], vitals=VitalSigns(spo2_percent=98))
            cls.router.run(warmup)

    def test_emergency_on_low_spo2(self) -> None:
        intake = Intake(
            age_years=40,
            sex="male",
            symptoms=["cough"],
            vitals=VitalSigns(spo2_percent=88),
        )
        result = self.__class__.router.run(intake)
        self.assertEqual(result.urgency, "emergency")
        self.assertTrue(result.guardrail_triggered)
        self.assertGreater(len(result.guardrail_reasons), 0)

    def test_urgent_for_keyword(self) -> None:
        intake = Intake(
            age_years=22,
            sex="female",
            symptoms=["persistent vomiting"],
            vitals=VitalSigns(spo2_percent=97),
        )
        result = self.__class__.router.run(intake)
        self.assertEqual(result.urgency, "urgent")
        self.assertFalse(result.guardrail_triggered)

    def test_chest_pain_is_at_least_urgent(self) -> None:
        intake = Intake(
            age_years=25,
            sex="male",
            symptoms=["chest pain"],
            vitals=VitalSigns(spo2_percent=97),
        )
        result = self.__class__.router.run(intake)
        self.assertIn(result.urgency, {"urgent", "emergency"})

    def test_remote_village_adjusts_next_step(self) -> None:
        intake = Intake(
            age_years=35,
            sex="female",
            symptoms=["shortness of breath"],
            vitals=VitalSigns(spo2_percent=88),
            environment="remote_village",
        )
        result = self.__class__.router.run(intake)
        self.assertIn("nurse-led stabilization", result.recommended_next_step.lower())

    def test_limited_access_region_adds_cost_aware_action(self) -> None:
        intake = Intake(
            age_years=46,
            sex="male",
            symptoms=["persistent vomiting"],
            vitals=VitalSigns(spo2_percent=96),
            environment="limited_access_region",
        )
        result = self.__class__.router.run(intake)
        combined_actions = " ".join(result.immediate_actions).lower()
        self.assertIn("lower-cost diagnostics", combined_actions)

    def test_patient_verification_and_history_fetch(self) -> None:
        intake = Intake(
            first_name="Anna",
            last_name="Kowalska",
            id_number="PL-998877",
            age_years=52,
            sex="female",
            symptoms=["shortness of breath"],
            vitals=VitalSigns(spo2_percent=95),
        )
        result = self.__class__.router.run(intake)
        self.assertTrue(result.patient_verified)
        self.assertTrue(result.patient_record_found)
        self.assertGreater(len(result.patient_history), 0)

    def test_mock_session_camera_and_abrasion_flow(self) -> None:
        workflow = SessionTriageWorkflow(router=self.__class__.router)
        session = PatientSessionInput(
            first_name="Anna",
            last_name="Kowalska",
            id_number="PL-998877",
            age_years=52,
            sex="female",
            patient_message="I have cough and a skin abrasion with minor bleeding.",
            camera_enabled=True,
            mock_followup_answers={"duration": "2 days"},
        )
        output = workflow.run(session)
        self.assertGreater(len(output.intake.interview_transcript), 0)
        self.assertIsNotNone(output.intake.camera_scene_description)
        self.assertGreater(len(output.intake.abrasion_findings), 0)
        self.assertEqual(output.intake.duration, "2 days")

    def test_guardrail_overrides_to_emergency_on_stroke_keyword(self) -> None:
        intake = Intake(
            age_years=60,
            sex="male",
            symptoms=["mild headache"],
            notes="Family reports sudden stroke symptoms and one-sided weakness.",
            vitals=VitalSigns(spo2_percent=97),
        )
        result = self.__class__.router.run(intake)
        self.assertEqual(result.urgency, "emergency")
        self.assertTrue(result.guardrail_triggered)
        reasons = " ".join(result.guardrail_reasons).lower()
        self.assertIn("stroke", reasons)

    def test_guardrail_precedence_over_standard_urgent_path(self) -> None:
        intake = Intake(
            age_years=44,
            sex="female",
            symptoms=["persistent vomiting", "severe chest pain"],
            vitals=VitalSigns(spo2_percent=96),
        )
        result = self.__class__.router.run(intake)
        self.assertEqual(result.urgency, "emergency")
        self.assertTrue(result.guardrail_triggered)

    def test_single_node_graph_run_from_symptoms_with_optional_image(self) -> None:
        assistant = ClinicalAssistantGraph(triage_router=self.__class__.router)
        output = assistant.run_from_symptoms(
            "I have shortness of breath and cough",
            age_years=50,
            sex="female",
            environment="remote_village",
            duration="3 days",
            image_path_or_url="https://example.com/chest-xray.png",
        )
        self.assertEqual(output.intake.scan_image_path, "https://example.com/chest-xray.png")
        self.assertGreater(len(output.intake.symptoms), 0)
        self.assertIsNotNone(output.triage.recommended_next_step)

    def test_conversation_flow_verification_and_followup_loop(self) -> None:
        from accolade_med_assistant.models.types import ConversationSessionInput

        assistant = ConversationAgentGraph(triage_router=self.__class__.router)
        session = ConversationSessionInput(
            first_name="Anna",
            last_name="Kowalska",
            id_number="PL-998877",
            age_years=50,
            sex="male",
            camera_estimated_age=22,
            camera_estimated_sex="female",
            verification_followup_answers={"identity_check": "ID confirmed with nurse."},
            patient_message="I have shortness of breath for 2 days.",
            duration="2 days",
            additional_questions=("Can you repeat the plan?", "How can I arrange transport?"),
        )
        output = assistant.run(session)
        joined_notes = " ".join(output.verification_notes).lower()
        self.assertIn("mismatch", joined_notes)
        self.assertGreater(len(output.dialogue), 1)


if __name__ == "__main__":
    unittest.main()
