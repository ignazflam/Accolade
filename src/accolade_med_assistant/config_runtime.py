from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("/Users/krzysztof/Documents/Accolade/data/input/app_config.json")


@dataclass(frozen=True)
class RuntimeSettings:
    environment: str = "standard"
    location_label: str = "unspecified"


def load_runtime_settings(config_path: Path | None = None) -> RuntimeSettings:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return RuntimeSettings()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return RuntimeSettings()

    environment = str(payload.get("environment", "standard")).strip() or "standard"
    location_label = str(payload.get("location_label", "unspecified")).strip() or "unspecified"
    return RuntimeSettings(environment=environment, location_label=location_label)
