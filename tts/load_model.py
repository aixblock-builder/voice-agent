import argparse
import json
from pathlib import Path


def load_config(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_default.json", help="Đường dẫn file JSON")
args = parser.parse_args()
config = load_config(args.config)

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