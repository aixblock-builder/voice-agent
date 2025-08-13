from __future__ import annotations
from typing import Any, Dict, Optional

from transformers import pipeline
import torch

from asr_plugin_base import AsrPlugin, register_asr_plugin

@register_asr_plugin("simple")
class SimplePipelineProvider(AsrPlugin):
    """
    Minimal provider that constructs an ASR pipeline directly from a model_id.
    Useful for models where transformers can auto-wire processor/tokenizer.
    """
    def build(self):
        model_id = self.cfg.get("model_id")
        if not model_id:
            raise ValueError("simple provider needs 'model_id'")

        # Device & dtype resolution
        device = self.cfg.get("device", 0 if torch.cuda.is_available() else "cpu")
        dtype_cfg = self.cfg.get("torch_dtype")
        if dtype_cfg is None:
            torch_dtype = torch.float16 if (isinstance(device, int) and device != "cpu" and torch.cuda.is_available()) else torch.float32
        else:
            torch_dtype = getattr(torch, dtype_cfg, dtype_cfg)

        pipeline_kwargs: Dict[str, Any] = dict(self.cfg.get("pipeline_kwargs", {}))
        if "token" in self.cfg:
            pipeline_kwargs["token"] = self.cfg["token"]
            pipeline_kwargs.setdefault("use_auth_token", self.cfg["token"])
        if "revision" in self.cfg:
            pipeline_kwargs["revision"] = self.cfg["revision"]
        if "trust_remote_code" in self.cfg:
            pipeline_kwargs["trust_remote_code"] = self.cfg["trust_remote_code"]
        if torch_dtype is not None:
            pipeline_kwargs["torch_dtype"] = torch_dtype

        asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            **pipeline_kwargs,
        )
        return asr
