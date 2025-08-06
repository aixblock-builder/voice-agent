from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)
import torch
model_id = "openai/whisper-large-v3"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = 0 if torch.cuda.is_available() else "cpu"
print("Loading Whisper model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded!")