from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class PatientRecord:
    first_name: str
    last_name: str
    id_number: str
    history: Sequence[str]
    scans: Sequence[str]


class LocalPatientRepository:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path or "/Users/krzysztof/Documents/Accolade/data/input/patients.json")

    def find_patient(self, first_name: str, last_name: str, id_number: str) -> Optional[PatientRecord]:
        payload = self._load_payload()
        for entry in payload.get("patients", []):
            if self._matches(entry, first_name, last_name, id_number):
                return PatientRecord(
                    first_name=str(entry.get("first_name", "")),
                    last_name=str(entry.get("last_name", "")),
                    id_number=str(entry.get("id_number", "")),
                    history=tuple(entry.get("history", [])),
                    scans=tuple(entry.get("scans", [])),
                )
        return None

    def _load_payload(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"patients": []}
        try:
            return json.loads(self.db_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"patients": []}

    def _matches(self, entry: dict[str, Any], first_name: str, last_name: str, id_number: str) -> bool:
        entry_id = str(entry.get("id_number", "")).strip()
        entry_first = str(entry.get("first_name", "")).strip().lower()
        entry_last = str(entry.get("last_name", "")).strip().lower()

        return (
            entry_id == id_number.strip()
            and entry_first == first_name.strip().lower()
            and entry_last == last_name.strip().lower()
        )
