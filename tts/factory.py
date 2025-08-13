from __future__ import annotations
import importlib
import pkgutil
from typing import Any
from tts_plugin_base import get_registry, get_plugin_class
import plugins
class TTSFactory:
    @staticmethod
    def _ensure_plugins_loaded() -> None:
        """Auto-discover & import tất cả module plugin trong realtimetts_plugins.
        Giải quyết lỗi Engines: [] khi các module (kokoro/xtts) chưa được import.
        """
        for m in pkgutil.iter_modules(plugins.__path__):
            if not m.ispkg:
                importlib.import_module(f"{plugins.__name__}.{m.name}")
                
    @staticmethod
    def engines():
        return list(get_registry().keys())

    @staticmethod
    def create(engine_name: str, **engine_kwargs: Any):
        cls = get_plugin_class(engine_name)
        if not cls:
            raise KeyError(f"Engine '{engine_name}' không tồn tại. Engines: {TTSFactory.engines()}")
        return cls(**engine_kwargs)