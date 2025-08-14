import asyncio
import base64
from typing import Optional, Dict, Any
from speech_to_text.plugin_loader import *
from speech_to_text.asr_base import create_plugin as create_asr_plugin

# Global storage for ASR plugins
active_plugins_asr: Dict[str, Any] = {}

def initialize_asr_plugin(agent_name: str, asr_config: Dict[str, Any]) -> bool:
    """Initialize ASR plugin for an agent"""
    try:
        asr_plugin = create_asr_plugin(
            asr_config["plugin_type"],
            **asr_config["config_model"]
        ).load()
        active_plugins_asr[agent_name] = asr_plugin
        print(f"[ASR] ASR plugin loaded for agent: {agent_name}")
        return True
    except Exception as e:
        print(f"[ASR] Failed to initialize ASR plugin for {agent_name}: {e}")
        return False

async def speech_to_text_with_plugin(
    audio_data: bytes, agent_name: str
) -> Optional[str]:
    """Convert speech to text using active ASR plugin"""
    try:
        # Get ASR plugin for this agent
        asr_plugin = active_plugins_asr.get(agent_name, active_plugins_asr.get(agent_name))
        print(active_plugins_asr)
        
        if not asr_plugin:
            print(f"[STT] No ASR plugin found for agent: {agent_name}")
            return None

        # Convert audio data to format expected by plugin
        # audio_file = convert_audio_to_wav(audio_data)

        # Run STT
        result = await asyncio.to_thread(asr_plugin.predict, audio_data)

        if isinstance(result, dict):
            return result.get("text", "")
        return str(result) if result else None

    except Exception as e:
        print(f"[STT] Error: {e}")
        return None

def convert_audio_to_wav(audio_data: bytes) -> str:
    """Convert audio data to WAV file for ASR plugin"""
    try:
        # Create temporary WAV file
        temp_file = f"/tmp/audio_{asyncio.get_event_loop().time()}.wav"

        # Assume audio_data is already in WAV format (16kHz, mono)
        # If not, you may need additional conversion
        with open(temp_file, "wb") as f:
            f.write(audio_data)

        return temp_file

    except Exception as e:
        print(f"[Audio] Conversion error: {e}")
        return None

def get_active_asr_plugins():
    """Get all active ASR plugins"""
    return active_plugins_asr

def cleanup_asr_plugin(agent_name: str) -> bool:
    """Clean up ASR plugin for an agent"""
    try:
        if agent_name in active_plugins_asr:
            del active_plugins_asr[agent_name]
            print(f"[ASR] Cleaned up ASR plugin for agent: {agent_name}")
            return True
        return False
    except Exception as e:
        print(f"[ASR] Failed to cleanup ASR plugin for {agent_name}: {e}")
        return False