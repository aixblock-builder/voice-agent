from __future__ import annotations
from typing import Any
from tts_plugin_base import BaseTTSPlugin, register_plugin

try:
    from RealtimeTTS import KokoroEngine
except Exception as e:  # pragma: no cover
    raise ImportError("Cần RealtimeTTS với KokoroEngine") from e

@register_plugin
class KokoroPlugin(BaseTTSPlugin):
    @classmethod
    def key(cls) -> str:
        return "kokoro"

    def build_engine(self) -> Any:
        return KokoroEngine(**self.engine_kwargs)