from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Dict, Type
import logging

try:
    from RealtimeTTS import TextToAudioStream
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Không import được RealtimeTTS. Cài đặt: pip install RealtimeTTS"
    ) from e


logger = logging.getLogger(__name__)

PLUGIN_REGISTRY: Dict[str, Type] = {}

def register_plugin(cls: Type) -> Type:
    key = getattr(cls, "key")()
    key = key.lower()
    PLUGIN_REGISTRY[key] = cls
    logger.info("Registered TTS plugin: %s", key)
    return cls

def get_registry() -> Dict[str, Type]:
    return dict(PLUGIN_REGISTRY)

def get_plugin_class(name: str):
    return PLUGIN_REGISTRY.get(name.lower())

class BaseTTSPlugin(ABC):
    """Giao diện chung cho plugin TTS."""

    @classmethod
    @abstractmethod
    def key(cls) -> str:
        """Tên plugin, ví dụ: 'kokoro', 'xtts'"""
        raise NotImplementedError

    def __init__(self, **engine_kwargs: Any) -> None:
        self.engine_kwargs = engine_kwargs
        self._engine = None

    @abstractmethod
    def build_engine(self) -> Any:
        """Trả về instance engine RealtimeTTS cụ thể."""
        raise NotImplementedError

    def get_engine(self) -> Any:
        if self._engine is None:
            self._engine = self.build_engine()
        return self._engine

    def create_stream(
        self,
        *,
        muted: bool = True,
        on_audio_stream_stop: Optional[Callable[[], None]] = None,
        **stream_kwargs: Any,
    ) -> TextToAudioStream:
        engine = self.get_engine()
        return TextToAudioStream(
            engine=engine,
            muted=muted,
            on_audio_stream_stop=on_audio_stream_stop,
            **stream_kwargs,
        )

    def speak(
        self,
        text: str,
        *,
        muted: bool = True,
        on_audio_stream_stop: Optional[Callable[[], None]] = None,
        stream_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TextToAudioStream:
        stream = self.create_stream(
            muted=muted,
            on_audio_stream_stop=on_audio_stream_stop,
            **(stream_kwargs or {}),
        )
        if hasattr(stream, "say"):
            stream.say(text)
        elif hasattr(stream, "speak"):
            stream.speak(text)
        elif hasattr(stream, "feed"):
            stream.feed(text)
        else:  # pragma: no cover
            raise RuntimeError("Không tìm thấy phương thức phát (say/speak/feed) trên TextToAudioStream")
        return stream