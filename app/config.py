from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    omnivoice_model: str = Field(default="k2-fsa/OmniVoice")
    omnivoice_device: Literal["auto", "cuda", "mps", "cpu"] = Field(default="auto")
    omnivoice_cuda_index: int = Field(default=0)
    omnivoice_dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    omnivoice_cache_dir: Optional[str] = Field(default=None)

    omnivoice_num_step: int = Field(default=32, ge=1, le=64)
    omnivoice_guidance_scale: float = Field(default=2.0, ge=0.0, le=10.0)
    omnivoice_t_shift: float = Field(default=0.1)
    omnivoice_denoise: bool = Field(default=True)
    omnivoice_postprocess_output: bool = Field(default=True)
    omnivoice_layer_penalty_factor: float = Field(default=5.0)
    omnivoice_position_temperature: float = Field(default=5.0)
    omnivoice_class_temperature: float = Field(default=0.0)

    omnivoice_prompt_cache_size: int = Field(default=32, ge=0, le=1024)
    omnivoice_voices_dir: str = Field(default="/voices")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    max_input_chars: int = Field(default=8000)
    default_response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3")

    @property
    def voices_path(self) -> Path:
        return Path(self.omnivoice_voices_dir)

    @property
    def resolved_device(self) -> str:
        import torch

        if self.omnivoice_device == "auto":
            if torch.cuda.is_available():
                return f"cuda:{self.omnivoice_cuda_index}"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if self.omnivoice_device == "cuda":
            return f"cuda:{self.omnivoice_cuda_index}"
        return self.omnivoice_device

    @property
    def torch_dtype(self):
        import torch

        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.omnivoice_dtype]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
