from __future__ import annotations
from typing import Dict, Any, Callable, Type
from abc import ABC, abstractmethod

PLUGIN_REGISTRY: Dict[str, Type["AsrPlugin"]] = {}

def register_asr_plugin(name: str):
    def deco(cls: Type["AsrPlugin"]):
        PLUGIN_REGISTRY[name] = cls
        return cls
    return deco

class AsrPlugin(ABC):
    """
    Base interface for providers that construct a Hugging Face ASR pipeline.
    Subclasses must implement build(self) -> pipeline.
    """
    def __init__(self, **kwargs: Any) -> None:
        self.cfg: Dict[str, Any] = kwargs

    @abstractmethod
    def build(self):
        """Return a transformers.pipeline configured for ASR."""
        raise NotImplementedError

def make_asr_pipe(plugin: str, **params: Any):
    if plugin not in PLUGIN_REGISTRY:
        raise KeyError(f"Unknown ASR plugin '{plugin}'. Available: {list(PLUGIN_REGISTRY.keys())}")
    cls = PLUGIN_REGISTRY[plugin]
    return cls(**params).build()
