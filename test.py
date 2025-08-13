import json
import os


config = {
  "provider": "whisper_seq2seq",
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
cfg_path = os.path.join("stt", "config.json")

if config is not None:
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[setup_app] Saved config to {cfg_path}")