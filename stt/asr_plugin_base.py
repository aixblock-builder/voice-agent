"""
ASR Plugin System - Hệ thống plugin cho Speech-to-Text
Hỗ trợ load models từ HuggingFace, Git và local files
"""

from __future__ import annotations
from typing import Dict, Any, Callable, Type, Optional, Union, List
from abc import ABC, abstractmethod
import os
import logging
from pathlib import Path
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import whisper
import librosa
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plugin registry
PLUGIN_REGISTRY: Dict[str, Type["AsrPlugin"]] = {}

def register_asr_plugin(name: str):
    """Decorator để đăng ký ASR plugin"""
    def deco(cls: Type["AsrPlugin"]):
        PLUGIN_REGISTRY[name] = cls
        return cls
    return deco

class AsrPlugin(ABC):
    """
    Base interface cho ASR plugins
    Hỗ trợ load models từ nhiều nguồn khác nhau
    """
    
    def __init__(self, **kwargs: Any) -> None:
        self.cfg: Dict[str, Any] = kwargs
        self.model = None
        self.processor = None
        self.pipeline = None
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Tự động detect device phù hợp"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model từ source được chỉ định"""
        raise NotImplementedError
    
    @abstractmethod
    def transcribe(self, audio: Union[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Transcribe audio thành text"""
        raise NotImplementedError
    
    @abstractmethod
    def build(self):
        """Build và return pipeline/model đã load"""
        raise NotImplementedError
    
    def preprocess_audio(self, audio: Union[str, np.ndarray], 
                        target_sr: int = 16000) -> np.ndarray:
        """Preprocess audio file"""
        if isinstance(audio, str):
            if not os.path.exists(audio):
                raise FileNotFoundError(f"Audio file not found: {audio}")
            audio_data, sr = librosa.load(audio, sr=target_sr)
        else:
            audio_data = audio
            
        return audio_data
    
    def get_info(self) -> Dict[str, Any]:
        """Trả về thông tin về plugin"""
        return {
            "plugin_name": self.__class__.__name__,
            "device": self.device,
            "config": self.cfg,
            "model_loaded": self.model is not None
        }

@register_asr_plugin("huggingface")
class HuggingFaceAsrPlugin(AsrPlugin):
    """Plugin cho models từ HuggingFace Hub"""
    
    def __init__(self, model_id: str = "openai/whisper-base", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.task = kwargs.get("task", "automatic-speech-recognition")
        self.load_model()
    
    def load_model(self) -> None:
        """Load model từ HuggingFace Hub"""
        try:
            logger.info(f"Loading model {self.model_id} from HuggingFace...")
            
            # Load với pipeline để đơn giản
            self.pipeline = pipeline(
                task=self.task,
                model=self.model_id,
                device=self.device if self.device != "mps" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                **self.cfg.get("pipeline_kwargs", {})
            )
            
            logger.info(f"Model {self.model_id} loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise
    
    def transcribe(self, audio: Union[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Transcribe audio using HuggingFace pipeline"""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            audio_data = self.preprocess_audio(audio)
            
            result = self.pipeline(
                audio_data,
                **kwargs
            )
            
            return {
                "text": result.get("text", ""),
                "confidence": result.get("confidence", None),
                "segments": result.get("chunks", []),
                "language": result.get("language", None)
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def build(self):
        """Return the loaded pipeline"""
        if self.pipeline is None:
            self.load_model()
        return self.pipeline

@register_asr_plugin("whisper")
class WhisperAsrPlugin(AsrPlugin):
    """Plugin cho OpenAI Whisper models"""
    
    def __init__(self, model_size: str = "base", **kwargs):
        super().__init__(**kwargs)
        self.model_size = model_size
        self.model_path = kwargs.get("model_path", None)  # Cho local model
        self.language = kwargs.get("language", None)
        self.load_model()
    
    def load_model(self) -> None:
        """Load Whisper model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading Whisper model from local path: {self.model_path}")
                self.model = whisper.load_model(self.model_path)
            else:
                logger.info(f"Loading Whisper model: {self.model_size}")
                self.model = whisper.load_model(
                    self.model_size,
                    device=self.device,
                    **self.cfg.get("model_kwargs", {})
                )
            
            logger.info(f"Whisper model {self.model_size} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio: Union[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Whisper có thể nhận file path trực tiếp hoặc numpy array
            if isinstance(audio, np.ndarray):
                # Whisper expects audio trong khoảng [-1, 1]
                if audio.max() > 1.0:
                    audio = audio / 32768.0  # Normalize từ int16
            
            # Whisper transcription options
            options = {
                "language": self.language,
                "task": kwargs.get("task", "transcribe"),  # transcribe hoặc translate
                "fp16": self.device == "cuda",
                **kwargs
            }
            
            result = self.model.transcribe(audio, **options)
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", None),
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"], 
                        "text": seg["text"].strip(),
                        "confidence": seg.get("avg_logprob", None)
                    }
                    for seg in result.get("segments", [])
                ],
                "confidence": None  # Whisper không trả confidence tổng
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise
    
    def build(self):
        """Return the loaded Whisper model"""
        if self.model is None:
            self.load_model()
        return self.model

@register_asr_plugin("git")
class GitAsrPlugin(AsrPlugin):
    """Plugin cho models từ Git repositories"""
    
    def __init__(self, repo_url: str, model_path: str = "", **kwargs):
        super().__init__(**kwargs)
        self.repo_url = repo_url
        self.model_path = model_path
        self.local_dir = kwargs.get("local_dir", "./downloaded_models")
        self.load_model()
    
    def _clone_repo(self) -> str:
        """Clone git repository"""
        import git
        
        repo_name = self.repo_url.split("/")[-1].replace(".git", "")
        target_path = os.path.join(self.local_dir, repo_name)
        
        if os.path.exists(target_path):
            logger.info(f"Repository already exists at {target_path}")
            return target_path
        
        logger.info(f"Cloning repository {self.repo_url}...")
        os.makedirs(self.local_dir, exist_ok=True)
        git.Repo.clone_from(self.repo_url, target_path)
        
        return target_path
    
    def load_model(self) -> None:
        """Load model từ Git repository"""
        try:
            repo_path = self._clone_repo()
            model_full_path = os.path.join(repo_path, self.model_path)
            
            # Thử load như HuggingFace model
            self.pipeline = pipeline(
                task="automatic-speech-recognition",
                model=model_full_path,
                device=self.device if self.device != "mps" else -1,
                **self.cfg.get("pipeline_kwargs", {})
            )
            
            logger.info(f"Git model loaded from {model_full_path}")
            
        except Exception as e:
            logger.error(f"Failed to load Git model: {e}")
            raise
    
    def transcribe(self, audio: Union[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Transcribe using Git model"""
        return HuggingFaceAsrPlugin.transcribe(self, audio, **kwargs)
    
    def build(self):
        """Return the loaded pipeline"""
        if self.pipeline is None:
            self.load_model()
        return self.pipeline

@register_asr_plugin("local")
class LocalAsrPlugin(AsrPlugin):
    """Plugin cho models từ local filesystem"""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.load_model()
    
    def load_model(self) -> None:
        """Load model từ local path"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            logger.info(f"Loading local model from {self.model_path}")
            
            self.pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model_path,
                device=self.device if self.device != "mps" else -1,
                **self.cfg.get("pipeline_kwargs", {})
            )
            
            logger.info(f"Local model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def transcribe(self, audio: Union[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Transcribe using local model"""
        return HuggingFaceAsrPlugin.transcribe(self, audio, **kwargs)
    
    def build(self):
        """Return the loaded pipeline"""
        if self.pipeline is None:
            self.load_model()
        return self.pipeline

# Factory function
def make_asr_pipe(plugin: str, **params: Any) -> AsrPlugin:
    """Factory function để tạo ASR plugin"""
    if plugin not in PLUGIN_REGISTRY:
        raise KeyError(f"Unknown ASR plugin '{plugin}'. Available: {list(PLUGIN_REGISTRY.keys())}")
    
    cls = PLUGIN_REGISTRY[plugin]
    return cls(**params)

def list_available_plugins() -> List[str]:
    """List tất cả plugins có sẵn"""
    return list(PLUGIN_REGISTRY.keys())

def get_plugin_info(plugin: str) -> Dict[str, Any]:
    """Lấy thông tin về plugin"""
    if plugin not in PLUGIN_REGISTRY:
        raise KeyError(f"Unknown plugin: {plugin}")
    
    cls = PLUGIN_REGISTRY[plugin]
    return {
        "name": plugin,
        "class": cls.__name__,
        "doc": cls.__doc__
    }

# # Example usage
# if __name__ == "__main__":
#     # Sử dụng Whisper plugin
#     print("Testing Whisper plugin...")
#     whisper_plugin = make_asr_pipe("whisper", model_size="base", language="vi")
    
#     # Test với file audio (thay đổi path cho phù hợp)
#     # result = whisper_plugin.transcribe("path/to/audio.wav")
#     # print(f"Transcription: {result['text']}")
    
#     # Sử dụng HuggingFace plugin
#     print("\nTesting HuggingFace plugin...")
#     hf_plugin = make_asr_pipe("huggingface", model_id="openai/whisper-small")
    
#     print(f"\nAvailable plugins: {list_available_plugins()}")
#     print(f"Whisper plugin info: {whisper_plugin.get_info()}")