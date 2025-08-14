# tts_base.py
from abc import ABC, abstractmethod
from typing import Dict, Type
import torch
import numpy as np
import soundfile as sf
from io import BytesIO

TTS_PLUGIN_REGISTRY: Dict[str, Type["TtsBase"]] = {}

def register_tts_plugin(name: str):
    def decorator(cls):
        TTS_PLUGIN_REGISTRY[name] = cls
        return cls
    return decorator

class TtsBase(ABC):
    def __init__(self, **kwargs):
        self.cfg = kwargs
        self.model = None
        self.device = self._get_device()
        self.sample_rate = kwargs.get("sample_rate", 22050)
    
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod 
    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        pass
    
    def save_audio(self, audio: np.ndarray, path: str):
        sf.write(path, audio, self.sample_rate)
    
    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        buffer = BytesIO()
        sf.write(buffer, audio, self.sample_rate, format='WAV')
        return buffer.getvalue()

def create_plugin(name: str, **kwargs) -> TtsBase:
    if name not in TTS_PLUGIN_REGISTRY:
        raise ValueError(f"TTS Plugin {name} not found")
    return TTS_PLUGIN_REGISTRY[name](**kwargs)