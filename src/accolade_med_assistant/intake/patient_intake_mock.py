from typing import List

from accolade_med_assistant.models.types import Intake, PatientSessionInput


class MockPatientIntakeAgent:
    """Mock pre-triage intake agent.

    Simulates:
    - short LLM interview with patient
    - camera scene description
    - abrasion scan findings
    """

    def build_intake(self, session: PatientSessionInput) -> Intake:
        transcript = self._mock_interview_transcript(session)
        symptoms = self._extract_symptoms(session.patient_message, transcript)
        camera_description = self._mock_camera_description(session) if session.camera_enabled else None
        abrasions = self._mock_abrasion_scan(session) if session.camera_enabled else []

        notes = session.patient_message.strip()
        if camera_description:
            notes = f"{notes} Camera: {camera_description}".strip()

        return Intake(
            age_years=session.age_years,
            sex=session.sex,
            symptoms=symptoms,
            first_name=session.first_name,
            last_name=session.last_name,
            id_number=session.id_number,
            duration=session.duration,
            notes=notes,
            interview_transcript=transcript,
            camera_scene_description=camera_description,
            abrasion_findings=abrasions,
            environment=session.environment,
        )

    def _mock_interview_transcript(self, session: PatientSessionInput) -> List[str]:
        base_questions = [
            "When did symptoms start?",
            "Are symptoms getting worse?",
            "Do you have fever, trouble breathing, chest pain, or bleeding?",
        ]
        answers = [
            f"Patient reported duration: {session.duration or 'unknown'}.",
            "Patient states symptoms are ongoing.",
            f"Patient initial statement: {session.patient_message or 'no details provided.'}",
        ]
        return [f"Q: {q} A: {a}" for q, a in zip(base_questions, answers)]

    def _extract_symptoms(self, message: str, transcript: List[str]) -> List[str]:
        text = f"{message} {' '.join(transcript)}".lower()
        keyword_map = {
            "shortness of breath": ("shortness of breath", "breathless", "difficulty breathing"),
            "chest pain": ("chest pain",),
            "fever": ("fever", "high temperature"),
            "cough": ("cough",),
            "abdominal pain": ("abdominal pain", "stomach pain"),
            "vomiting": ("vomiting", "vomit"),
            "bleeding": ("bleeding", "blood loss"),
            "abrasion": ("abrasion", "scrape", "wound", "cut"),
        }

        detected: List[str] = []
        for normalized, variants in keyword_map.items():
            if any(term in text for term in variants):
                detected.append(normalized)

        return detected or ["general malaise"]

    def _mock_camera_description(self, session: PatientSessionInput) -> str:
        text = session.patient_message.lower()
        if "cough" in text or "breath" in text:
            return "Patient appears fatigued with mild increased breathing effort."
        if "wound" in text or "cut" in text or "abrasion" in text:
            return "Visible superficial skin injury noted on exposed area."
        return "Patient visible in stable seated posture; no obvious distress detected."

    def _mock_abrasion_scan(self, session: PatientSessionInput) -> List[str]:
        text = session.patient_message.lower()
        findings: List[str] = []
        if any(token in text for token in ("abrasion", "scrape", "cut", "wound")):
            findings.append("Possible superficial abrasion detected")
        if "bleeding" in text:
            findings.append("Possible active bleeding signal detected")
        return findings
