from language_model.llm_base import LlmBase, register_llm_plugin
import torch

@register_llm_plugin("qwen")
class QwenPlugin(LlmBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load(self, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        source = self._get_model_source()
        self.tokenizer = AutoTokenizer.from_pretrained(source, **self.config_tokenizer)
        device = self._get_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float16
        if device == "cuda":
            self.pipeline = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, device_map="auto", **self.config_pipeline)
        else:
            self.pipeline = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, device_map="cpu", **self.config_pipeline)
        return self
    
    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.pipeline.generate(
            inputs["input_ids"],
            max_length=kwargs.get("max_length", 512),
            num_beams=kwargs.get("num_beams", 1),
            do_sample=kwargs.get("do_sample", False),
            temperature=kwargs.get("temperature", 1.0)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def save_model(self, path: str):
        self.pipeline.save_pretrained(path)
        self.tokenizer.save_pretrained(path)