from dataclasses import dataclass, field
from typing import Sequence

from accolade_med_assistant.models.types import EnvironmentType


@dataclass(frozen=True)
class TriageConfig:
    emergency_keywords: Sequence[str] = field(
        default_factory=lambda: (
            "severe chest pain",
            "difficulty breathing",
            "shortness of breath",
            "unconscious",
            "seizure",
            "stroke",
            "one-sided weakness",
            "heavy bleeding",
            "coughing blood",
            "suicidal",
        )
    )
    urgent_keywords: Sequence[str] = field(
        default_factory=lambda: (
            "chest pain",
            "chest pressure",
            "pain in chest",
            "fever",
            "persistent vomiting",
            "dehydration",
            "worsening cough",
            "blood in urine",
            "abdominal pain",
            "pregnant with pain",
        )
    )
    emergency_spo2_threshold: int = 90
    urgent_spo2_threshold: int = 94


@dataclass(frozen=True)
class EnvironmentProfile:
    name: EnvironmentType
    guidance_note: str
    escalation_note: str
    care_constraints: Sequence[str]


def default_environment_profiles() -> dict[EnvironmentType, EnvironmentProfile]:
    return {
        "standard": EnvironmentProfile(
            name="standard",
            guidance_note="Standard referral pathways are assumed to be available.",
            escalation_note="If red flags are present, proceed directly to hospital emergency care.",
            care_constraints=(),
        ),
        "remote_village": EnvironmentProfile(
            name="remote_village",
            guidance_note=(
                "Remote setting with no physician nearby; nurse-led stabilization and monitoring are primary."
            ),
            escalation_note=(
                "If emergency signs are present, arrange fastest available transport while continuing stabilization."
            ),
            care_constraints=(
                "No immediate physician access.",
                "Advanced imaging may not be available locally.",
                "Care plan should prioritize practical bedside monitoring steps.",
            ),
        ),
        "limited_access_region": EnvironmentProfile(
            name="limited_access_region",
            guidance_note=(
                "Public/private service access may be delayed or limited by cost and local availability."
            ),
            escalation_note=(
                "For urgent cases, prioritize same-day SOR/NPL pathways and nearest available diagnostics."
            ),
            care_constraints=(
                "Some diagnostics (for example MRI) may be unavailable in the city.",
                "Budget constraints should be considered when proposing tests.",
                "Prefer stepwise diagnostics and feasible referrals.",
            ),
        ),
    }
