import os

import torch
from huggingface_hub import HfFolder
from transformers import pipeline

# Đặt token của bạn vào đây
hf_token = os.getenv("HF_TOKEN", "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")
# Lưu token vào local
HfFolder.save_token(hf_token)

from huggingface_hub import login


login(token=hf_token)


def _load():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        print("CUDA is available.")

        _model = pipeline(
            "text-generation",
            model="Qwen/Qwen3-1.7B",
            torch_dtype=dtype,
            device_map="auto",
            max_new_tokens=256,
        )
    else:
        print("No GPU available, using CPU.")
        _model = pipeline(
            "text-generation",
            model="Qwen/Qwen3-1.7B",
            device_map="cpu",
            max_new_tokens=256,
        )


_load()
