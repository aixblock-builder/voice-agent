import os
import subprocess
import threading
import shutil

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
    pip_exec = os.path.join(venv_path, "bin", "pip")
    requirements_file = os.path.join(folder, "requirements.txt")
    if os.path.exists(requirements_file):
        subprocess.run([pip_exec, "install", "-r", requirements_file], check=True)

    cmd = f"cd {folder} && venv/bin/python load_model.py"
    return subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def load_model(folder):
    cmd = f"cd {folder} && venv/bin/python load_model.py"
    return subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


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


def stream_output(pipe):
    for line in iter(pipe.readline, ""):
        line = line.strip()
        if line:
            print(line)
    pipe.close()


def run_stt_app_func():
    global stt_proc
    try:
        stt_proc = run_app(stt_folder)
        # Tạo thread để đọc stdout và stderr song song
        stdout_thread = threading.Thread(target=stream_output, args=(stt_proc.stdout,))
        stderr_thread = threading.Thread(target=stream_output, args=(stt_proc.stderr,))
        stdout_thread.start()
        stderr_thread.start()
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


def run_tts_app_func():
    global tts_proc
    try:
        tts_proc = run_app(tts_folder)
        # Tạo thread để đọc stdout và stderr song song
        stdout_thread = threading.Thread(target=stream_output, args=(tts_proc.stdout,))
        stderr_thread = threading.Thread(target=stream_output, args=(tts_proc.stderr,))
        stdout_thread.start()
        stderr_thread.start()
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


stt_thread = threading.Thread(target=run_stt_app_func, daemon=True)
tts_thread = threading.Thread(target=run_tts_app_func, daemon=True)
