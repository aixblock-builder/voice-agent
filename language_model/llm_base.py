from abc import ABC, abstractmethod
import os
from typing import Dict, Type
import torch

LLM_PLUGIN_REGISTRY: Dict[str, Type["LlmBase"]] = {}

def register_llm_plugin(name: str):
    def decorator(cls):
        LLM_PLUGIN_REGISTRY[name] = cls
        return cls
    return decorator

class LlmBase(ABC):
    def __init__(self, **kwargs):
        self.model_id = kwargs['model_id']
        self.config_tokenizer = kwargs['config_tokenizer'] if "config_tokenizer" in kwargs else {}
        self.config_pipeline = kwargs['config_pipeline'] if "config_pipeline" in kwargs else {}
        self.pipeline = None
        self.tokenizer = None
        self.processor = None
        self.device = self._get_device()
    
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _get_model_source(self):
        model_name = self.model_id.split("/")[-1]
        local_path = f"./data/checkpoint/{model_name}"
        return local_path if os.path.exists(f"{local_path}/config.json") else self.model_id

    @abstractmethod 
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod 
    def save_model(self, path: str):
        pass

    @abstractmethod 
    def load(self, **kwargs):
        pass
    

def create_plugin(name: str, **kwargs) -> LlmBase:
    if name not in LLM_PLUGIN_REGISTRY:
        raise ValueError(f"LLM Plugin {name} not found")
    return LLM_PLUGIN_REGISTRY[name](**kwargs)