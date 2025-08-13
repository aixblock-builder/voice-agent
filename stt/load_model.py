from factory import build_asr_pipe
from pathlib import Path
import json
import argparse

def load_config(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_default.json", help="Đường dẫn file JSON")
args = parser.parse_args()
config = load_config(args.config)

build_asr_pipe(config)

print("Model loaded!")