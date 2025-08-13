from __future__ import annotations
from typing import Any, Dict

import torch
from transformers import (
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoFeatureExtractor,
    pipeline,
)

from asr_plugin_base import AsrPlugin, register_asr_plugin

@register_asr_plugin("custom_components")
class CustomComponentsProvider(AsrPlugin):
    """
    Provide separate model/tokenizer/feature_extractor ids for maximum control.
    Works with either CTC or Seq2Seq models depending on 'architecture'.
    """
    def build(self):
        arch = (self.cfg.get("architecture") or "seq2seq").lower()
        model_id = self.cfg.get("model_id")
        if not model_id:
            raise ValueError("custom_components needs 'model_id'")

        tok_id = self.cfg.get("tokenizer_id", model_id)
        fe_id = self.cfg.get("feature_extractor_id", model_id)

        device = self.cfg.get("device", 0 if torch.cuda.is_available() else "cpu")
        if "torch_dtype" in self.cfg and self.cfg["torch_dtype"] is not None:
            torch_dtype = getattr(torch, self.cfg["torch_dtype"], self.cfg["torch_dtype"])
        else:
            torch_dtype = torch.float16 if (isinstance(device, int) and device != "cpu" and torch.cuda.is_available()) else torch.float32

        token = self.cfg.get("token")
        revision = self.cfg.get("revision")
        trust_remote_code = self.cfg.get("trust_remote_code", False)

        if arch == "ctc":
            ModelClass = AutoModelForCTC
        else:
            ModelClass = AutoModelForSpeechSeq2Seq

        model = ModelClass.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(tok_id, token=token, revision=revision, trust_remote_code=trust_remote_code)
        feature_extractor = AutoFeatureExtractor.from_pretrained(fe_id, token=token, revision=revision, trust_remote_code=trust_remote_code)

        pipeline_kwargs: Dict[str, Any] = dict(self.cfg.get("pipeline_kwargs", {}))
        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            **pipeline_kwargs,
        )
        return asr
