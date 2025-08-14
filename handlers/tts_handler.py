import asyncio
import base64
import threading
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from text_to_speech.plugin_loader import *
from text_to_speech.tts_base import create_plugin as create_tts_plugin

# Global TTS instance management
_global_tts_instances = {}
_instance_lock = threading.Lock()
_tts_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts_pool")

# Active TTS plugins
active_plugins_tts: Dict[str, Any] = {}

def initialize_tts_plugin(agent_name: str, tts_config: Dict[str, Any]) -> bool:
    """Initialize TTS plugin for an agent"""
    try:
        tts_plugin = create_tts_plugin(
            tts_config["plugin_type"],
            **tts_config["config_model"]
        ).load()
        active_plugins_tts[agent_name] = tts_plugin
        set_tts_instance(agent_name, tts_plugin)  # Lưu vào global
        
        print(f"[TTS] TTS loaded and ready for agent: {agent_name}")
        return True
    except Exception as e:
        print(f"[TTS] Failed to initialize TTS plugin for {agent_name}: {e}")
        return False

def get_tts_instance(agent_name: str):
    """Lấy TTS instance từ global storage"""
    return _global_tts_instances.get(agent_name)

def set_tts_instance(agent_name: str, tts_instance):
    """Set TTS instance vào global storage với thread safety"""
    with _instance_lock:
        _global_tts_instances[agent_name] = tts_instance
        print(f"[TTS] Global instance set for agent: {agent_name}")

async def text_to_speech(text: str, agent_name: str) -> Optional[str]:
    """Convert text to speech using pre-loaded instance"""
    def _synthesize():
        # Lấy instance từ global storage trước
        tts_plugin = get_tts_instance(agent_name)
        
        # Fallback về active_plugins_tts nếu không có trong global
        if not tts_plugin:
            tts_plugin = active_plugins_tts.get(agent_name)
            if tts_plugin:
                # Lưu vào global cho lần sau
                set_tts_instance(agent_name, tts_plugin)
        
        if not tts_plugin:
            print(f"[TTS] No TTS plugin found for agent: {agent_name}")
            return None
        
        # Synthesize (model đã load sẵn)
        audio = tts_plugin.synthesize(text)
        buffer = tts_plugin.audio_to_bytes(audio)
        return base64.b64encode(buffer).decode("ascii")
    
    try:
        loop = asyncio.get_event_loop()
        audio_b64 = await loop.run_in_executor(_tts_thread_pool, _synthesize)
        return audio_b64
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return None

def get_active_tts_plugins():
    """Get all active TTS plugins"""
    return active_plugins_tts

def get_global_tts_instances():
    """Get all global TTS instances"""
    return _global_tts_instances

def cleanup_tts_plugin(agent_name: str) -> bool:
    """Clean up TTS plugin for an agent"""
    try:
        cleanup_success = False
        
        # Clean up from active plugins
        if agent_name in active_plugins_tts:
            del active_plugins_tts[agent_name]
            cleanup_success = True
        
        # Clean up from global instances
        with _instance_lock:
            if agent_name in _global_tts_instances:
                del _global_tts_instances[agent_name]
                cleanup_success = True
        
        if cleanup_success:
            print(f"[TTS] Cleaned up TTS plugin for agent: {agent_name}")
        
        return cleanup_success
    except Exception as e:
        print(f"[TTS] Failed to cleanup TTS plugin for {agent_name}: {e}")
        return False

def shutdown_tts_thread_pool():
    """Shutdown the TTS thread pool"""
    try:
        _tts_thread_pool.shutdown(wait=True)
        print("[TTS] Thread pool shutdown successfully")
    except Exception as e:
        print(f"[TTS] Error shutting down thread pool: {e}")