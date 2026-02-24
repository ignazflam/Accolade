from pathlib import Path
from typing import Any, Optional
import io
import os


class LocalLLMClient:
    """Abstraction point for a local text model.

    Supported backends:
    - fallback: static offline-safe summary text
    - medgemma: local transformers pipeline for medical summarization
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-1.5-4b-it",
        backend: str = "fallback",
        device: str = "mps",
        dtype: str = "float32",
        max_new_tokens: int = 220,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.backend = os.getenv("ACCOLADE_LLM_BACKEND", backend).lower()
        self.device = os.getenv("ACCOLADE_LLM_DEVICE", device)
        self.dtype = os.getenv("ACCOLADE_LLM_DTYPE", dtype).lower()
        self.max_new_tokens = max_new_tokens
        env_local_only = os.getenv("ACCOLADE_LLM_LOCAL_ONLY", str(local_files_only)).lower()
        self.local_files_only = env_local_only in {"1", "true", "yes", "on"}
        self._medgemma_pipe: Any = None
        self._medgemma_init_failed = False

    def summarize(self, prompt: str, scan_image_path: Optional[str] = None) -> Optional[str]:
        if self.backend == "medgemma":
            medgemma_summary = self._summarize_with_medgemma(prompt, scan_image_path)
            if medgemma_summary:
                return medgemma_summary

        # Offline-safe fallback. Keeps POC running without heavyweight dependencies.
        if not prompt.strip():
            return None
        return (
            "Preliminary support summary generated without a connected local LLM. "
            "Set ACCOLADE_LLM_BACKEND=medgemma to use local MedGemma summarization."
        )

    def _summarize_with_medgemma(self, prompt: str, scan_image_path: Optional[str]) -> Optional[str]:
        if self._medgemma_init_failed:
            return None
        try:
            pipe = self._get_medgemma_pipeline()
        except Exception:
            self._medgemma_init_failed = True
            return None

        content = []
        image = self._load_image(scan_image_path) if scan_image_path else None
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            result = pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        except Exception:
            return None

        return self._extract_text(result)

    def _get_medgemma_pipeline(self):
        if self._medgemma_pipe is not None:
            return self._medgemma_pipe

        import torch
        from transformers import pipeline

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(self.dtype, torch.float32)

        self._medgemma_pipe = pipeline(
            "image-text-to-text",
            model=self.model_name,
            dtype=resolved_dtype,
            device=self.device,
            local_files_only=self.local_files_only,
        )
        return self._medgemma_pipe

    def _load_image(self, scan_image_path: str):
        from PIL import Image
        import requests

        if scan_image_path.startswith(("http://", "https://")):
            response = requests.get(
                scan_image_path,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X)"},
                timeout=30,
            )
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")

        image_path = Path(scan_image_path).expanduser()
        return Image.open(image_path).convert("RGB")

    def _extract_text(self, result: Any) -> Optional[str]:
        if not result:
            return None

        first = result[0] if isinstance(result, list) else result
        generated = first.get("generated_text") if isinstance(first, dict) else None

        if isinstance(generated, str):
            text = generated.strip()
            return text or None

        if isinstance(generated, list):
            for message in reversed(generated):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        text = content.strip()
                        return text or None

        return None
