import asyncio
import json
import httpx
from utils.socket_manager import ConnectionManager
import subprocess
import shutil
import os

TTS_HTTP_URL = "http://localhost:1006/tts-only"

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

async def get_audio_from_tts_service(client_id: str, text: str, manager: ConnectionManager):
    """
    Hàm này gọi đến TTS service qua HTTP và stream phản hồi về cho client.
    """
    params = {
        "client_id": client_id,
        "text": text,
        "speaker_wav": "example_female.wav",
        "language": "en"
    }
    
    try:
        # Sử dụng httpx.AsyncClient để thực hiện streaming request
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", TTS_HTTP_URL, params=params) as response:
                print(f"Gateway: Bắt đầu nhận stream từ TTS cho client '{client_id}'")
                
                # Đọc từng dòng trong stream (mỗi dòng là một JSON)
                async for line in response.aiter_lines():
                    if line:
                        try:
                            # Parse JSON và gửi về frontend qua WebSocket
                            json_data = json.loads(line)
                            json_data["type"] = "audio"
                            await manager.send_json_to_client(json_data, client_id)
                        except json.JSONDecodeError:
                            print(f"Gateway: Lỗi khi parse JSON từ TTS: {line}")
                            # Gửi tin nhắn lỗi về cho client
                            await manager.send_json_to_client({"error": "Invalid JSON from TTS."}, client_id)
    
    except asyncio.CancelledError:
        print(f"[TTS] Stream bị hủy giữa chừng cho client {client_id}")
        raise
                            
    except httpx.RequestError as e:
        print(f"Gateway: Lỗi khi gọi đến TTS service: {e}")
        # Gửi tin nhắn lỗi về cho client
        await manager.send_json_to_client({"error": "TTS service is unavailable."}, client_id)