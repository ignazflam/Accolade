from pathlib import Path
from typing import Any, Optional
import io
import os
import platform
import sys


class LocalLLMClient:
    """Abstraction point for a local text model.

    Supported backends:
    - fallback: static offline-safe summary text
    - medgemma: local HF transformers multimodal pipeline
    - deepseek: local HF transformers text-generation pipeline
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-1.5-4b-it",
        backend: str = "fallback",
        device: str = "auto",
        dtype: str = "float32",
        max_new_tokens: int = 220,
        local_files_only: bool = False,
    ) -> None:
        self.last_error: Optional[str] = None
        root_dir = Path(__file__).resolve().parents[3]
        self.hf_token_file = Path(
            os.getenv("ACCOLADE_HF_TOKEN_FILE", str(root_dir / ".secrets" / "hf_token.txt"))
        ).expanduser()
        self.hf_token = self._load_hf_token()
        self.model_name = model_name
        self.backend = os.getenv("ACCOLADE_LLM_BACKEND", backend).lower()
        self.device = self._resolve_device(os.getenv("ACCOLADE_LLM_DEVICE", device))
        self.dtype = os.getenv("ACCOLADE_LLM_DTYPE", dtype).lower()
        self.max_new_tokens = int(os.getenv("ACCOLADE_LLM_MAX_NEW_TOKENS", str(max_new_tokens)))
        self.max_time_seconds = float(os.getenv("ACCOLADE_LLM_MAX_TIME_SECONDS", "30"))
        env_local_only = os.getenv("ACCOLADE_LLM_LOCAL_ONLY", str(local_files_only)).lower()
        self.local_files_only = env_local_only in {"1", "true", "yes", "on"}
        self._medgemma_pipe: Any = None
        self._medgemma_init_failed = False
        self._deepseek_pipe: Any = None
        self._deepseek_init_failed = False

    def summarize(self, prompt: str, scan_image_path: Optional[str] = None) -> Optional[str]:
        return self.generate_text(prompt, scan_image_path=scan_image_path)

    def generate_text(self, prompt: str, scan_image_path: Optional[str] = None) -> Optional[str]:
        if self.backend == "medgemma":
            medgemma_summary = self._summarize_with_medgemma(prompt, scan_image_path)
            if medgemma_summary:
                return medgemma_summary
        elif self.backend == "deepseek":
            deepseek_summary = self._summarize_with_deepseek(prompt)
            if deepseek_summary:
                return deepseek_summary

        # Offline-safe fallback. Keeps POC running without heavyweight dependencies.
        if not prompt.strip():
            return None
        error_suffix = f" Last error: {self.last_error}" if self.last_error else ""
        return (
            "Preliminary support summary generated without a connected local LLM. "
            "Ensure Hugging Face model access is configured and model files are available locally."
            f"{error_suffix}"
        )

    def _summarize_with_deepseek(self, prompt: str) -> Optional[str]:
        if self._deepseek_init_failed:
            return None
        try:
            pipe = self._get_deepseek_pipeline()
        except Exception as exc:
            self._deepseek_init_failed = True
            self._log_error("deepseek", "init", exc)
            return None

        try:
            result = pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                max_time=self.max_time_seconds,
                do_sample=False,
                return_full_text=False,
            )
        except Exception as exc:
            self._log_error("deepseek", "generate", exc)
            return None
        return self._extract_generated_text(result)

    def _summarize_with_medgemma(self, prompt: str, scan_image_path: Optional[str]) -> Optional[str]:
        if self._medgemma_init_failed:
            return None
        try:
            pipe = self._get_medgemma_pipeline()
        except Exception as exc:
            self._medgemma_init_failed = True
            self._log_error("medgemma", "init", exc)
            return None

        content = []
        try:
            image = self._load_image(scan_image_path) if scan_image_path else None
        except Exception as exc:
            self._log_error("medgemma", "image_load", exc)
            return None
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            result = pipe(
                messages,
                max_new_tokens=self.max_new_tokens,
                max_time=self.max_time_seconds,
                do_sample=False,
            )
        except Exception as exc:
            self._log_error("medgemma", "generate", exc)
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

        self._medgemma_pipe = self._build_pipeline_with_device_fallback(
            task="image-text-to-text",
            model=self.model_name,
            dtype=resolved_dtype,
            token=self.hf_token,
        )
        return self._medgemma_pipe

    def _get_deepseek_pipeline(self):
        if self._deepseek_pipe is not None:
            return self._deepseek_pipe

        import torch
        from transformers import pipeline

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(self.dtype, torch.float32)
        model_id = (
            os.getenv("ACCOLADE_DECIDER_MODEL")
            or os.getenv("ACCOLADE_TEXT_MODEL")
            or os.getenv("ACCOLADE_DEEPSEEK_MODEL")
            or self.model_name
        )

        self._deepseek_pipe = self._build_pipeline_with_device_fallback(
            task="text-generation",
            model=model_id,
            dtype=resolved_dtype,
            token=self.hf_token,
        )
        return self._deepseek_pipe

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

    def _extract_generated_text(self, result: Any) -> Optional[str]:
        if not result:
            return None
        first = result[0] if isinstance(result, list) else result
        if isinstance(first, dict):
            text = str(first.get("generated_text", "")).strip()
            return text or None
        return None

    def _log_error(self, backend: str, stage: str, exc: Exception) -> None:
        message = f"{backend}.{stage} failed: {type(exc).__name__}: {exc}"
        self.last_error = message
        print(f"[LocalLLMClient] {message}", file=sys.stderr)

    def _load_hf_token(self) -> Optional[str]:
        env_token = (
            os.getenv("ACCOLADE_HF_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
        )
        if env_token and env_token.strip():
            return env_token.strip()

        try:
            if self.hf_token_file.exists():
                content = self.hf_token_file.read_text(encoding="utf-8")
                for line in content.splitlines():
                    token = line.strip()
                    if token and not token.startswith("#"):
                        return token
        except Exception as exc:
            self._log_error("huggingface", "token_load", exc)
        return None

    def _build_pipeline_with_device_fallback(self, task: str, model: str, dtype, token: Optional[str]):
        from transformers import pipeline

        try:
            return pipeline(
                task,
                model=model,
                dtype=dtype,
                device=self.device,
                token=token,
            )
        except Exception as exc:
            should_retry_cpu = self.device == "mps" and (
                "mps backend is supported on macos 14.0+" in str(exc).lower()
                or "mps" in str(exc).lower()
            )
            if not should_retry_cpu:
                raise
            self._log_error("llm", "device_fallback", exc)
            return pipeline(
                task,
                model=model,
                dtype=dtype,
                device="cpu",
                token=token,
            )

    def _resolve_device(self, requested: str) -> str:
        requested_norm = (requested or "").strip().lower()
        if requested_norm and requested_norm != "auto":
            return requested_norm

        try:
            import torch
        except Exception:
            return "cpu"

        if not torch.backends.mps.is_available():
            return "cpu"

        mac_ver = platform.mac_ver()[0]
        try:
            major = int(mac_ver.split(".")[0]) if mac_ver else 0
        except Exception:
            major = 0
        if major < 14:
            return "cpu"
        return "mps"
