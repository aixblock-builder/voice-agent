from __future__ import annotations
from typing import Any, Dict, Union
import json

from asr_plugin_base import make_asr_pipe  # loads from in-memory registry (plugins must be imported)

# Ensure plugins are registered when this module is imported
import plugins.simple_pipeline  # noqa: F401
import plugins.whisper_seq2seq  # noqa: F401
import plugins.custom_components  # noqa: F401

def build_asr_pipe(config: Union[str, Dict[str, Any]]):
    """
    Return an ASR transformers pipeline from either a dict or a JSON file path.
    Expected shape:
      {
        "plugin": "whisper_seq2seq" | "simple" | "custom_components",
        "params": { ... }  # plugin-specific
      }
    """
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = config

    plugin = cfg.get("plugin", "whisper_seq2seq")
    params = cfg.get("params", {})
    return make_asr_pipe(plugin, **params)
