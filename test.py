import asyncio
import websockets
import soundfile as sf
import base64
import json

WS_URL = "ws://136.59.129.136:33475/conversation"
AUDIO_FILE = "test.mp3"

async def listen_server(ws):
    """Lắng nghe phản hồi server liên tục và lưu audio nếu có"""
    try:
        async for message in ws:
            data = json.loads(message)
            print("Server:", data)

            # Kiểm tra nếu server gửi audio
            if data.get("type") == "audio" and "audio_event" in data:
                audio_b64 = data["audio_event"].get("audio_base_64")
                if audio_b64:
                    # Decode base64 sang bytes
                    audio_bytes = base64.b64decode(audio_b64)

                    # Lưu trực tiếp bytes thành WAV
                    with open("server_audio.wav", "wb") as f:
                        f.write(audio_bytes)
                    print("Saved audio from server: server_audio.wav")
    except websockets.ConnectionClosed:
        print("Connection closed by server")

async def main():
    async with websockets.connect(WS_URL) as ws:
        # Gửi init
        init_msg = {
            "type": "conversation_initiation_client_data",
            "agent_name": "my_agent",
            "conversation_config_override": {
                "agent": {
                    "prompt": {"prompt": "You are a helpful assistant"},
                    "first_message": "Hello! How can I help you today?"
                }
            }
        }
        await ws.send(json.dumps(init_msg))
        print("Sent conversation initiation")

        # Task lắng nghe server song song
        asyncio.create_task(listen_server(ws))

        # Đọc toàn bộ file WAV/MP3
        data, samplerate = sf.read(AUDIO_FILE, dtype='int16')
        byte_data = data.tobytes()

        # Encode toàn bộ audio thành base64
        chunk_b64 = base64.b64encode(byte_data).decode("utf-8")

        # Gửi audio 1 chunk tổng thể
        audio_msg = {
            "type": "user_audio_chunk",
            "user_audio_chunk": chunk_b64
        }
        await ws.send(json.dumps(audio_msg))
        print("Sent entire audio as 1 chunk")

        # Chờ vài giây để server phản hồi (hoặc implement event khác)
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
