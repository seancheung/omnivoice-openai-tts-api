from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class SpeechRequest(BaseModel):
    """OpenAI-compatible `/v1/audio/speech` request (voice cloning only)."""

    model: Optional[str] = Field(default=None, description="Accepted for OpenAI compatibility; ignored.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice id matching a file pair in the voices directory.")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    language: Optional[str] = Field(default=None, description="Language name or ISO code.")
    duration: Optional[float] = Field(
        default=None, ge=0.5, le=120.0,
        description="Fixed output duration in seconds; overrides speed when set.",
    )
    num_step: Optional[int] = Field(default=None, ge=4, le=64)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=10.0)


class DesignRequest(BaseModel):
    """Non-OpenAI `/v1/audio/design` request using OmniVoice's voice-design mode."""

    input: str = Field(..., description="Text to synthesize.")
    instruct: str = Field(..., description="Voice design attributes (e.g. 'female, british accent').")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    language: Optional[str] = Field(default=None)
    duration: Optional[float] = Field(default=None, ge=0.5, le=120.0)
    num_step: Optional[int] = Field(default=None, ge=4, le=64)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=10.0)


class VoiceInfo(BaseModel):
    id: str
    preview_url: str
    prompt_text: str


class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str
    device: Optional[str] = None
    dtype: Optional[str] = None
    sample_rate: Optional[int] = None
