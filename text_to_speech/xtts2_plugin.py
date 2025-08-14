# xtts2_plugin.py
from .tts_base import TtsBase, register_tts_plugin
import torch
import numpy as np
from typing import Union, Optional
import librosa

@register_tts_plugin("xtts2")
class XTTS2Plugin(TtsBase):
    def __init__(self, model_path: str = None, language: str = "en", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.language = language
        self.speaker_wav = kwargs.get("speaker_wav", None)
        self.sample_rate = 24000  # XTTS2 default
        
    def load(self):
        try:
            from TTS.api import TTS
            
            # Use local model or download default
            if self.model_path:
                self.model = TTS(model_path=self.model_path, gpu=self.device == "cuda")
            else:
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=self.device == "cuda")
            
            return self
        except ImportError:
            raise ImportError("TTS not installed. pip install TTS")
    
    def synthesize(self, text: str, speaker_wav: Optional[str] = None, language: Optional[str] = None, **kwargs) -> np.ndarray:
        if self.model is None:
            self.load()
            
        selected_speaker = speaker_wav or self.speaker_wav
        selected_language = language or self.language
        
        if not selected_speaker:
            raise ValueError("speaker_wav required for XTTS2")
        
        # Generate audio
        audio = self.model.tts(
            text=text,
            speaker_wav=selected_speaker,
            language=selected_language,
            **kwargs
        )
        
        # Convert to numpy array
        if isinstance(audio, list):
            audio = np.array(audio)
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        return audio.astype(np.float32)
    
    def clone_voice(self, text: str, reference_audio: str, **kwargs) -> np.ndarray:
        """Clone voice from reference audio"""
        return self.synthesize(text, speaker_wav=reference_audio, **kwargs)

# Usage example  
if __name__ == "__main__":
    from tts_base import create_tts_plugin
    
    plugin = create_tts_plugin("xtts2", language="en", speaker_wav="reference.wav")
    audio = plugin.synthesize("Hello, this is voice cloning with XTTS2!")
    plugin.save_audio(audio, "cloned_output.wav")