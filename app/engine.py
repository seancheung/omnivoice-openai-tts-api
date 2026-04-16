from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import numpy as np
import torch
from omnivoice import OmniVoice

log = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, settings):
        self.settings = settings

        device = settings.resolved_device
        dtype = settings.torch_dtype
        if device == "cpu" and dtype == torch.float16:
            log.warning("cpu device detected; overriding dtype float16 -> float32")
            dtype = torch.float32

        if settings.omnivoice_cache_dir:
            os.environ.setdefault("HF_HOME", settings.omnivoice_cache_dir)

        log.info(
            "loading OmniVoice model=%s device=%s dtype=%s",
            settings.omnivoice_model, device, dtype,
        )
        self.model = OmniVoice.from_pretrained(
            settings.omnivoice_model, device_map=device, dtype=dtype,
        )
        self.device = device
        self.dtype_str = str(dtype).replace("torch.", "")
        self.sample_rate = int(self.model.sampling_rate)
        self._lock = asyncio.Lock()

        self._prompt_cache: dict[tuple, Any] = {}
        self._prompt_cache_order: list[tuple] = []
        self._prompt_cache_max = int(settings.omnivoice_prompt_cache_size)

    # ------------------------------------------------------------------
    # voice clone prompt caching
    # ------------------------------------------------------------------
    def _get_or_build_prompt(self, wav_path: str, prompt_text: str, mtime: float):
        key = (wav_path, mtime, prompt_text)
        hit = self._prompt_cache.get(key)
        if hit is not None:
            try:
                self._prompt_cache_order.remove(key)
            except ValueError:
                pass
            self._prompt_cache_order.append(key)
            return hit

        log.info("building voice clone prompt for %s", wav_path)
        vcp = self.model.create_voice_clone_prompt(
            ref_audio=wav_path, ref_text=prompt_text, preprocess_prompt=True,
        )
        if self._prompt_cache_max > 0:
            self._prompt_cache[key] = vcp
            self._prompt_cache_order.append(key)
            while len(self._prompt_cache_order) > self._prompt_cache_max:
                old = self._prompt_cache_order.pop(0)
                self._prompt_cache.pop(old, None)
        return vcp

    def _gen_kwargs(
        self,
        *,
        num_step: Optional[int],
        guidance_scale: Optional[float],
    ) -> dict:
        s = self.settings
        return dict(
            num_step=num_step if num_step is not None else s.omnivoice_num_step,
            guidance_scale=(
                guidance_scale if guidance_scale is not None else s.omnivoice_guidance_scale
            ),
            t_shift=s.omnivoice_t_shift,
            denoise=s.omnivoice_denoise,
            postprocess_output=s.omnivoice_postprocess_output,
            layer_penalty_factor=s.omnivoice_layer_penalty_factor,
            position_temperature=s.omnivoice_position_temperature,
            class_temperature=s.omnivoice_class_temperature,
        )

    @staticmethod
    def _unwrap(audios) -> np.ndarray:
        if isinstance(audios, list):
            if not audios:
                raise RuntimeError("OmniVoice returned empty audio list")
            wav = audios[0]
        else:
            wav = audios
        wav = np.asarray(wav)
        return np.ascontiguousarray(wav.astype(np.float32, copy=False))

    # ------------------------------------------------------------------
    # inference entrypoints
    # ------------------------------------------------------------------
    async def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio: str,
        ref_text: str,
        ref_mtime: float,
        language: Optional[str] = None,
        duration: Optional[float] = None,
        speed: float = 1.0,
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> np.ndarray:
        kwargs = self._gen_kwargs(num_step=num_step, guidance_scale=guidance_scale)

        async with self._lock:
            audios = await asyncio.to_thread(
                self._run_clone, text, ref_audio, ref_text, ref_mtime,
                language, duration, speed, kwargs,
            )
        return self._unwrap(audios)

    def _run_clone(self, text, ref_audio, ref_text, ref_mtime, language, duration, speed, kwargs):
        vcp = self._get_or_build_prompt(ref_audio, ref_text, ref_mtime)
        call_kwargs = dict(kwargs)
        if language is not None:
            call_kwargs["language"] = language
        if duration is not None:
            call_kwargs["duration"] = duration
        else:
            call_kwargs["speed"] = speed
        return self.model.generate(text=text, voice_clone_prompt=vcp, **call_kwargs)

    async def synthesize_design(
        self,
        text: str,
        *,
        instruct: str,
        language: Optional[str] = None,
        duration: Optional[float] = None,
        speed: float = 1.0,
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> np.ndarray:
        kwargs = self._gen_kwargs(num_step=num_step, guidance_scale=guidance_scale)

        async with self._lock:
            audios = await asyncio.to_thread(
                self._run_design, text, instruct, language, duration, speed, kwargs,
            )
        return self._unwrap(audios)

    def _run_design(self, text, instruct, language, duration, speed, kwargs):
        call_kwargs = dict(kwargs)
        if language is not None:
            call_kwargs["language"] = language
        if duration is not None:
            call_kwargs["duration"] = duration
        else:
            call_kwargs["speed"] = speed
        return self.model.generate(text=text, instruct=instruct, **call_kwargs)
