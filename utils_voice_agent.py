import json
import os
import subprocess
import threading
import shutil
import queue
import time
import contextlib
from typing import Optional

stt_proc = None
tts_proc = None
stt_folder = "stt"
tts_folder = "tts"


def ensure_portaudio_installed():
    if shutil.which("apt-get") is None:
        print(
            "apt-get không có sẵn (không phải Debian/Ubuntu?) -> bỏ qua cài đặt portaudio19-dev."
        )
        return

    # Dùng sudo nếu không chạy với quyền root
    sudo = [] if hasattr(os, "geteuid") and os.geteuid() == 0 else ["sudo"]

    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"

    cmds = [
        sudo + ["apt-get", "update"],
        sudo + ["apt-get", "install", "-y", "portaudio19-dev"],
    ]

    for cmd in cmds:
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)


def setup_app(folder: str, config: Optional[dict] = None):
    cfg_path = os.path.join(folder, "config.json")
    if config is not None:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"[setup_app] Saved config to {cfg_path}")
    # 2. Tạo venv nếu chưa có
    venv_path = os.path.join(folder, "venv")
    if not os.path.exists(venv_path):
        subprocess.run(f"python3 -m venv venv", shell=True, cwd=folder, check=True)

    # 3. Cài requirements
    requirements_file = os.path.join(folder, "requirements.txt")
    if os.path.exists(requirements_file):
        subprocess.run(f"venv/bin/pip install -r requirements.txt", shell=True, cwd=folder, check=True)
    for attempt in range(2):
        try:
            load_model(folder)
            break
        except Exception as e:
            if attempt == 0:
                shutil.rmtree(os.path.join(folder, "models"), ignore_errors=True)
                print("Load tts model failed, retrying after cleanup...")
                time.sleep(2)
            else:
                raise


def load_model(folder):
    load_model_file = os.path.join(folder, "load_model.py")
    if os.path.exists(load_model_file):
        cmd = f"cd {folder} && venv/bin/python load_model.py --config config.json"
        subprocess.run(
            cmd,
            shell=True,
            check=True,
        )
        print(f"Hoàn thành load model từ {load_model_file}")



def run_app(folder: str):
    cmd = f"cd {folder} && venv/bin/python app.py --config config.json"
    return subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def stream_output(pipe, q):
    try:
        for line in iter(pipe.readline, ''):
            q.put(line)
    finally:
        pipe.close()

def run_tts_app_func(config: Optional[dict] = None):
    setup_app(tts_folder, config)
    global tts_proc
    q = queue.Queue()
    try:
        tts_proc = run_app(tts_folder)
        # Tạo thread để đọc stdout và stderr song song
        stdout_thread = threading.Thread(target=stream_output, args=(tts_proc.stdout, q))
        stderr_thread = threading.Thread(target=stream_output, args=(tts_proc.stderr, q))
        stdout_thread.start()
        stderr_thread.start()
        while True:
            try:
                line = q.get(timeout=0.1)
                print(line, end='')
            except queue.Empty:
                if tts_proc.poll() is not None and q.empty():
                    break
        ret = tts_proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        print(f"Process exited with code {ret}")

    except Exception as e:
        print(e)
        if tts_proc and tts_proc.poll() is None:
            tts_proc.terminate()
            tts_proc.wait(timeout=5)
        if tts_proc:
            print(f"Process returned with code: {tts_proc.returncode}")

# TTS (tương tự nếu bạn có tts_proc)
def stop_tts_app():
    with contextlib.suppress(Exception):
        if tts_proc and tts_proc.poll() is None:
            tts_proc.terminate()
            try:
                tts_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tts_proc.kill()
