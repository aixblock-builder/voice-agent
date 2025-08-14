# whisper_plugin.py
from .asr_base import AsrBase, register_plugin
import whisper
import numpy as np
from typing import Union
import io
import base64

@register_plugin("whisper")
class WhisperPlugin(AsrBase):
    def __init__(self, model_size: str = "base", **kwargs):
        super().__init__(**kwargs)
        self.model_size = model_size
        self.language = kwargs.get("language", None)
        
    def load(self):
        self.model = whisper.load_model(self.model_size, device=self.device)
        return self
    
    def predict(self, audio: Union[str, np.ndarray, bytes]) -> str:
        if self.model is None:
            self.load()
        
        # Handle file path, numpy array, or base64 bytes
        if isinstance(audio, str):
            if audio.startswith('data:') or len(audio) > 255:  # Base64 string
                # Decode base64 to bytes
                audio_bytes = base64.b64decode(audio)
                # Convert bytes to numpy array (assuming 16kHz, 16-bit PCM)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # File path - load and preprocess
                audio_data = self.preprocess(audio)
        elif isinstance(audio, bytes):
            # Raw bytes - convert to numpy array
            audio_data = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            # Assume it's numpy array or file path from converter
            audio_data = self.preprocess(audio)
            
        result = self.model.transcribe(
            audio_data,
            language=self.language,
            fp16=self.device == "cuda"
        )
        return result["text"].strip()