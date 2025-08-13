import asyncio
import websockets
import soundfile as sf
import base64
import json

WS_URL = "ws://136.59.129.136:33401/conversation"
AUDIO_FILE = "test.mp3"

async def main():
    async with websockets.connect(WS_URL) as ws:
        # B1: Gửi init
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

        # Nhận phản hồi init
        msg = await ws.recv()
        print("Received:", msg)

        # Đọc file WAV
        data, samplerate = sf.read(AUDIO_FILE, dtype='int16')
        byte_data = data.tobytes()

        # Lấy 1 chunk đầu tiên
        chunk_size = 3200  # bytes
        chunk = byte_data[:chunk_size]
        chunk_b64 = base64.b64encode(chunk).decode("utf-8")

        # Gửi chunk audio
        audio_msg = {
            "type": "user_audio_chunk",
            "user_audio_chunk": chunk_b64
        }
        await ws.send(json.dumps(audio_msg))
        print("Sent first audio chunk")

        # Nhận phản hồi server
        try:
            while True:
                resp = await asyncio.wait_for(ws.recv(), timeout=5)
                print("Received:", resp)
        except asyncio.TimeoutError:
            print("No more messages from server.")

if __name__ == "__main__":
    asyncio.run(main())
