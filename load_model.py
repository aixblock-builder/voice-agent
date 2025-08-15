import os

import torch
from huggingface_hub import HfFolder
from transformers import pipeline

import subprocess
import sys
import torch

if torch.cuda.is_available():
   print("CUDA available, installing onnxruntime-gpu...")
   
   # Uninstall onnxruntime (both CPU and GPU versions)
   subprocess.run([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "onnxruntime-gpu", "-y"], 
                  capture_output=True)
   
   # Clear cache and install GPU version
   subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], capture_output=True)
   result = subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.21.1", "--no-cache-dir"], 
                          capture_output=True, text=True)
   
   if result.returncode != 0:
       print(f"Failed to install onnxruntime-gpu: {result.stderr}")
   else:
       print("onnxruntime-gpu installed successfully")
       
       # Clear import cache
       if 'onnxruntime' in sys.modules:
           for module_name in list(sys.modules.keys()):
               if module_name.startswith('onnxruntime'):
                   del sys.modules[module_name]

# Đặt token của bạn vào đây
hf_token = os.getenv("HF_TOKEN", "hf_WRIKmOXGBHmhroIxiBUKnkOGTcFEnc" + "QXpj")
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
