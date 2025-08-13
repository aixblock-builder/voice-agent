import os
import subprocess
import threading
import shutil
import queue
import time
import contextlib

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


def setup_app(repo: str, folder: str):
    # 1. Clone repo
    if not os.path.exists(folder):
        subprocess.run(
            f"git clone https://github.com/{repo} {folder}", shell=True, check=True
        )

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
        cmd = f"cd {folder} && venv/bin/python load_model.py"
        subprocess.run(
            cmd,
            shell=True,
            check=True,
        )
        print(f"Hoàn thành load model từ {load_model_file}")



def run_app(folder: str):
    cmd = f"cd {folder} && venv/bin/python app.py"
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


def run_stt_app_func(model):
    setup_app(model, stt_folder)
    global stt_proc
    q = queue.Queue()
    try:
        stt_proc = run_app(stt_folder)
        # Tạo thread để đọc stdout và stderr song song
        stdout_thread = threading.Thread(target=stream_output, args=(stt_proc.stdout, q))
        stderr_thread = threading.Thread(target=stream_output, args=(stt_proc.stderr, q))
        stdout_thread.start()
        stderr_thread.start()
        while True:
            try:
                line = q.get(timeout=0.1)
                print(line, end='')
            except queue.Empty:
                if stt_proc.poll() is not None and q.empty():
                    break
        ret = stt_proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        print(f"Process exited with code {ret}")
    except Exception as e:
        print("Runner error:", e)
        if stt_proc and stt_proc.poll() is None:
            stt_proc.terminate()
            stt_proc.wait(timeout=5)
        if stt_proc:
            print(f"Process returned with code: {stt_proc.returncode}")


def run_tts_app_func(model):
    setup_app(model, tts_folder)
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

# STT
def stop_stt_app():
    with contextlib.suppress(Exception):
        if stt_proc and stt_proc.poll() is None:
            stt_proc.terminate()
            try:
                stt_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stt_proc.kill()

# TTS (tương tự nếu bạn có tts_proc)
def stop_tts_app():
    with contextlib.suppress(Exception):
        if tts_proc and tts_proc.poll() is None:
            tts_proc.terminate()
            try:
                tts_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tts_proc.kill()
