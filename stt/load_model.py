from factory import build_asr_pipe

config_sample = {
  "plugin": "whisper_seq2seq",
  "params": {
    "model_id": "openai/whisper-large-v3",
    "language": "en",
    "low_cpu_mem_usage": True,
    "use_safetensors": True,
    "model_kwargs": {
      "language": "en"
    }
  }
}

build_asr_pipe(config_sample)

print("Model loaded!")