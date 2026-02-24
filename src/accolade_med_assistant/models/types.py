from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

EnvironmentType = Literal["standard", "remote_village", "limited_access_region"]


@dataclass
class VitalSigns:
    temperature_c: Optional[float] = None
    heart_rate_bpm: Optional[int] = None
    spo2_percent: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None


@dataclass
class Intake:
    age_years: Optional[int]
    sex: Optional[str]
    symptoms: Sequence[str]
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    id_number: Optional[str] = None
    duration: Optional[str] = None
    notes: str = ""
    interview_transcript: Sequence[str] = field(default_factory=tuple)
    camera_scene_description: Optional[str] = None
    abrasion_findings: Sequence[str] = field(default_factory=tuple)
    vitals: VitalSigns = field(default_factory=VitalSigns)
    scan_findings: Optional[str] = None
    scan_image_path: Optional[str] = None
    environment: EnvironmentType = "standard"


@dataclass
class TriageResult:
    urgency: str
    guardrail_triggered: bool
    guardrail_reasons: Sequence[str]
    patient_verified: bool
    patient_record_found: bool
    patient_history: Sequence[str]
    relevant_scans: Sequence[str]
    rationale: Sequence[str]
    immediate_actions: Sequence[str]
    recommended_next_step: str
    assistant_summary: str


@dataclass
class PatientSessionInput:
    age_years: Optional[int]
    sex: Optional[str]
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    id_number: Optional[str] = None
    environment: EnvironmentType = "standard"
    patient_message: str = ""
    scan_image_path: Optional[str] = None
    scan_findings: Optional[str] = None
    duration: Optional[str] = None
    camera_enabled: bool = True
    mock_followup_answers: dict[str, str] = field(default_factory=dict)


@dataclass
class ConversationSessionInput(PatientSessionInput):
    camera_estimated_age: Optional[int] = None
    camera_estimated_sex: Optional[str] = None
    verification_followup_answers: dict[str, str] = field(default_factory=dict)
    additional_questions: Sequence[str] = field(default_factory=tuple)
    use_medgemma: bool = False


@dataclass
class ConversationRunOutput:
    intake: Intake
    triage: TriageResult
    verification_notes: Sequence[str]
    medgemma_called: bool
    medgemma_decision_reason: str
    medgemma_feedback: Optional[str]
    dialogue: Sequence[str]
