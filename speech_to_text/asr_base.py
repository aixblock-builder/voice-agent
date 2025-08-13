# asr_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Type
import torch
import librosa
import numpy as np

PLUGIN_REGISTRY: Dict[str, Type["AsrBase"]] = {}

def register_plugin(name: str):
    def decorator(cls):
        PLUGIN_REGISTRY[name] = cls
        return cls
    return decorator

class AsrBase(ABC):
    def __init__(self, **kwargs):
        self.cfg = kwargs
        self.model = None
        self.device = self._get_device()
    
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod 
    def predict(self, audio: Union[str, np.ndarray]) -> str:
        pass
    
    def preprocess(self, audio: Union[str, np.ndarray], sr: int = 16000) -> np.ndarray:
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=sr)
        return audio

def create_plugin(name: str, **kwargs) -> AsrBase:
    if name not in PLUGIN_REGISTRY:
        raise ValueError(f"Plugin {name} not found")
    return PLUGIN_REGISTRY[name](**kwargs)