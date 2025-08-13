# config = {
#     "engine_name": "coqui",
#     "engine_kwargs": {
#         "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
#         "specific_model": "v2.0.2",
#         "voices_path": "./speakers",
#     }
# }

config = {
    "engine_name": "kokoro",
    "engine_kwargs": {
        "voice": "af_heart",
    }
}

def load_model():
    from factory import TTSFactory
    try:
        TTSFactory._ensure_plugins_loaded()
    except Exception:
        pass
    plugin = TTSFactory.create(config["engine_name"], **config["engine_kwargs"])
    engine = plugin.get_engine()
    print("[Load Model] Engine ready.")
    try:
        engine.shutdown()
        print("[Load Model] Engine shutdown.")
    except Exception as e:
        print("[Load Model] shutdown() failed:", e)

if __name__ == "__main__":
    load_model()