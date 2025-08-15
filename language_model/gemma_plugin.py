from language_model.llm_base import LlmBase, register_llm_plugin
import torch
from transformers import (
    Gemma3nForConditionalGeneration,
    AutoProcessor,
)


@register_llm_plugin("gemma")
class GemmaPlugin(LlmBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, **kwargs):
        source = self._get_model_source()
        self.processor = AutoProcessor.from_pretrained(source, **self.config_processor)
        device = self._get_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float16
        if device == "cuda":
            self.pipeline = Gemma3nForConditionalGeneration.from_pretrained(
                source,
                device_map="auto",
                torch_dtype=dtype,
            ).eval()
        else:
            self.pipeline = Gemma3nForConditionalGeneration.from_pretrained(
                source,
                device_map="cpu",
                torch_dtype=dtype,
            ).eval()
        return self

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.pipeline.device)
        input_len = inputs["input_ids"].shape[-1]

        # conduct text completion
        generated_ids = self.pipeline.generate(
            **inputs,
            do_sample=False
        )

        output_ids = generated_ids[0][input_len:]
        generated_text = self.processor.decode(output_ids, skip_special_tokens=True).strip("\n")

        return generated_text

