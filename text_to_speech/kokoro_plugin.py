# kokoro_plugin.py
import os
import requests
import numpy as np
from pathlib import Path
from .tts_base import TtsBase, register_tts_plugin

try:
    from kokoro_onnx import Kokoro
except ImportError:
    raise ImportError("Please install kokoro-onnx: pip install kokoro-onnx")

@register_tts_plugin("kokoro")
class KokoroPlugin(TtsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = kwargs.get("model_path", "kokoro-v0_19.onnx")
        self.voices_path = kwargs.get("voices_path", "voices.bin")
        self.voice = kwargs.get("voice", "af")  # Default voice
        self.speed = kwargs.get("speed", 1.0)  # Tốc độ phù hợp cho GPU
        self.lang = kwargs.get("lang", "en-us")
        self.kokoro = None
        self.sample_rate = 24000  # Kokoro sample rate
    
    def _download_file(self, url: str, path: str):
        """Download file if not exists"""
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    def load(self):
        """Load Kokoro model with GPU optimization"""
        # Download model files if needed
        model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
        voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
        
        self._download_file(model_url, self.model_path)
        self._download_file(voices_url, self.voices_path)
        
        # Set GPU provider for better performance
        os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        
        self.kokoro = Kokoro(
            model_path=self.model_path,
            voices_path=self.voices_path
        )
        print(f"Kokoro loaded on {self.device}")
    
    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        """Synthesize speech from text"""
        if self.kokoro is None:
            self.load()
        
        voice = kwargs.get("voice", self.voice)
        speed = kwargs.get("speed", self.speed)
        lang = kwargs.get("lang", self.lang)
        
        audio, _ = self.kokoro.create(
            text=text,
            voice=voice,
            speed=speed,
            lang=lang,
            trim=True  # Trim silence for better quality
        )
        
        return audio.astype(np.float32)
    
    def get_available_voices(self):
        """Get list of available voices"""
        if self.kokoro is None:
            self.load()
        return self.kokoro.get_voices()
