from __future__ import annotations
from typing import Any, Dict

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from asr_plugin_base import AsrPlugin, register_asr_plugin

@register_asr_plugin("whisper_seq2seq")
class WhisperSeq2SeqProvider(AsrPlugin):
    """
    Explicitly load AutoModelForSpeechSeq2Seq + AutoProcessor, then build a pipeline.
    Mirrors the structure in the user's sample.
    """
    def build(self):
        model_id = self.cfg.get("model_id", "openai/whisper-large-v3")

        # Resolve device
        if "device" in self.cfg:
            device = self.cfg["device"]
        else:
            device = 0 if torch.cuda.is_available() else "cpu"

        # Resolve dtype
        if "torch_dtype" in self.cfg and self.cfg["torch_dtype"] is not None:
            torch_dtype = getattr(torch, self.cfg["torch_dtype"], self.cfg["torch_dtype"])
        else:
            torch_dtype = torch.float16 if (isinstance(device, int) and device != "cpu" and torch.cuda.is_available()) else torch.float32

        low_cpu_mem_usage = self.cfg.get("low_cpu_mem_usage", True)
        use_safetensors = self.cfg.get("use_safetensors", True)

        model_kwargs: Dict[str, Any] = dict(self.cfg.get("model_kwargs", {}))
        if "language" in self.cfg and self.cfg["language"]:
            # Keep language in model_kwargs for generation if user desires
            model_kwargs.setdefault("language", self.cfg["language"])

        # Auth / misc
        token = self.cfg.get("token")
        revision = self.cfg.get("revision")
        trust_remote_code = self.cfg.get("trust_remote_code", False)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_safetensors=use_safetensors,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(
            model_id,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        # Optional: allow external tokenizer/feature_extractor IDs
        tok = processor.tokenizer
        fe = processor.feature_extractor

        # Build pipeline
        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tok,
            feature_extractor=fe,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs,
        )
        return asr
