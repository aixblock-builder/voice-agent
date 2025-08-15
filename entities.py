from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union

class ActionRequest(BaseModel):
    command: str
    params: Dict[str, Any]
    doc_file_urls: Optional[Union[str, List[str]]] = None
    session_id: Optional[str] = None
    use_history: Optional[bool] = True

class ASRConfig(BaseModel):
    plugin_type: str  # "whisper", "huggingface", etc
    config_model: Dict[str, Any]  # model-specific config

class ConfigLlmModel(BaseModel):
    model_id: str
    config_tokenizer: Dict[str, Any] = {}  # tokenizer-specific config
    config_pipeline: Dict[str, Any] = {}  # pipeline-specific config
    config_processor: Dict[str, Any] = {}  # processor-specific config

class LLMConfig(BaseModel):
    plugin_type: str  # "gpt", "llama", "qwen", etc
    config_model: ConfigLlmModel

class TTSEngineConfig(BaseModel):
    engine_name: str  # "kokoro", "coqui", etc
    engine_kwargs: Dict[str, Any]  # engine-specific config

class TTSConfig(BaseModel):
    plugin_type: str  # "xtts2", "kokoro", etc
    config_model: Dict[str, Any]  # model-specific config

class InitAgentRequest(BaseModel):
    name: str
    agent_name: str
    endpoint: str
    auth_token: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    memoryConnection: Optional[Dict[str, Any]] = None
    storageConnection: Optional[Dict[str, Any]] = None
    asr_config: Optional[ASRConfig] = None
    tts_config: Optional[TTSConfig] = None
    llm_config: Optional[LLMConfig] = None