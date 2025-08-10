# server.py
import asyncio
# import json
from audio_processor import AudioProcessor
from socket_utils import ConnectionManager
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)
from vad_utils import process_frame, SAMPLE_RATE, FRAME_SIZE
import uuid
from llm_client import LLMClient, handle_llm_response, llm_tts_pipeline
# from tts_client import TTSClient
import httpx
from fastapi_proxy_lib.fastapi.app import reverse_http_app
from fastapi.openapi.utils import get_openapi

manager = ConnectionManager()

app = FastAPI(
    title="MyModel API",
    openapi_url="/swagger.json",
    docs_url="/swagger",
)

LLM_WS_URL = "wss://localhost:3000/ws/stream-token"
# TTS_WS_URL = "wss://localhost:8002/ws/generate-audio"

# Khởi tạo cả 2 client với các hàm callback tương ứng
llm_client = LLMClient(LLM_WS_URL, manager=manager, on_response_callback=handle_llm_response)
# tts_client = TTSClient(TTS_WS_URL, on_json_callback=forward_tts_response_to_client)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(llm_client.connect_and_listen())
    # asyncio.create_task(tts_client.connect_and_listen())

# THÊM HÀM NÀY
@app.on_event("shutdown")
async def shutdown_event():
    """
    Hàm này được gọi khi ứng dụng dừng để đóng các kết nối.
    """
    print("Bắt đầu tắt ứng dụng, đóng các kết nối WebSocket...")
    await llm_client.close()
    # await tts_client.close()
    print("✅ Đã đóng tất cả kết nối an toàn.")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Load model (Hugging Face Whisper large-v3) ---
model_id = "openai/whisper-large-v3"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = 0 if torch.cuda.is_available() else "cpu"

print("Loading Whisper model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded!")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs={"language": "en"}
)

FRAMES_PER_BUFFER = 10
VAD_SPEECH_THRESHOLD = 0.6
SILENCE_TIMEOUT_SECONDS = 1.5
MAX_RECORDING_SECONDS = 30
CHUNK_SIZE_BYTES = FRAME_SIZE * 2 * FRAMES_PER_BUFFER
SILENCE_FRAMES_THRESHOLD = int((SILENCE_TIMEOUT_SECONDS * SAMPLE_RATE) / FRAME_SIZE)
MAX_RECORDING_FRAMES = int((MAX_RECORDING_SECONDS * SAMPLE_RATE) / FRAME_SIZE)
BYTES_PER_SAMPLE = 2
client_audio_connections = {}

services_to_merge = {
    "llm": "http://localhost:8001",
    "tts": "http://localhost:8002",
}

def custom_openapi():
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # đảm bảo có sẵn các trường cần thiết
    openapi_schema.setdefault("components", {}).setdefault("schemas", {})
    openapi_schema.setdefault("paths", {})
    openapi_schema.setdefault("tags", [])

    HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}

    def ensure_tag(tag_name: str):
        """Thêm tag vào openapi_schema['tags'] nếu chưa có."""
        if not any(t.get("name") == tag_name for t in openapi_schema["tags"]):
            openapi_schema["tags"].append({"name": tag_name})

    # -----------------------------------------------------------------
    #  Hợp nhất schema của từng micro-service & gán tag theo prefix
    # -----------------------------------------------------------------
    for prefix, base_url in services_to_merge.items():
        ensure_tag(prefix)
        try:
            resp = httpx.get(f"{base_url}/openapi.json")
            resp.raise_for_status()
            service_schema = resp.json()

            # --- paths ---
            for path, path_item in service_schema.get("paths", {}).items():
                new_path = f"/{prefix}{path}"
                openapi_schema["paths"][new_path] = path_item

                # gắn tag cho từng HTTP method
                for method, op in path_item.items():
                    if method in HTTP_METHODS:
                        op["tags"] = [prefix] + [
                            t for t in op.get("tags", []) if t != prefix
                        ]

            # --- components.schemas ---
            openapi_schema["components"]["schemas"].update(
                service_schema.get("components", {}).get("schemas", {})
            )

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            # fallback nếu service không truy cập được
            print(f"[WARN] Cannot fetch schema of '{prefix}': {e}")
            openapi_schema["paths"][f"/{prefix}/service_unavailable"] = {
                "get": {
                    "summary": f"Service '{prefix}' is unavailable",
                    "description": f"Could not connect to {base_url} to fetch API details.",
                    "tags": [prefix],
                    "responses": {"503": {"description": "Service Unavailable"}},
                }
            }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.get("/", tags=["stt"])
async def demo_interface():
    return FileResponse("index.html")

@app.get("/ui-test", tags=["stt"])
async def demo_interface_test():
    return FileResponse("index_test.html")

audio_processor = AudioProcessor(pipe=asr_pipe)

@app.websocket("/ws/audio_v2_test")
async def websocket_endpoint_test(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    try:
        while True:
            try:
                data = await websocket.receive_json()
                text = data.get("text", "").strip()
                if not text:
                    continue
                print(f"Received text: {text}")
                # 🔥 HUỶ pipeline cũ (nếu đang chạy)
                if manager.cancel_pipeline(client_id, llm_client.cleanup_fn):
                    # Thông báo client dừng audio nếu cần
                    await manager.send_json_to_client({"type": "control", "event": "interrupt"}, client_id)

                # 🔥 TẠO pipeline mới
                task = asyncio.create_task(
                    llm_tts_pipeline(client_id, text, llm_client, manager)
                )
                manager.set_pipeline_task(client_id, task, llm_client.cleanup_fn)
            except Exception as e:
                raise e
    except WebSocketDisconnect:
        print("Client disconnected")
        await manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.disconnect(client_id)
    finally:
        print(f"Closing connection for client {client_id}.")
        await manager.disconnect(client_id)


# --- WebSocket endpoint ---
@app.websocket("/ws/audio_v2")
async def websocket_endpoint_v2(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    audio_buffer = bytearray()
    recording_buffer = []
    is_recording = False
    silence_counter = 0
    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)

                if len(audio_buffer) < CHUNK_SIZE_BYTES:
                    continue

                chunk_bytes = audio_buffer[:CHUNK_SIZE_BYTES]
                del audio_buffer[:CHUNK_SIZE_BYTES]

                frames = []
                speech_frame_count = 0
                frame_byte_size = FRAME_SIZE * BYTES_PER_SAMPLE
                for i in range(0, len(chunk_bytes), frame_byte_size):
                    frame_data = chunk_bytes[i:i + frame_byte_size]
                    if len(frame_data) < frame_byte_size:
                        continue # Bỏ qua frame cuối nếu không đủ byte

                    # Gọi hàm xử lý mới, thuần túy
                    is_speech, waveform = process_frame(frame_data)
                    
                    if waveform is not None:
                        frames.append(waveform)
                        if is_speech:
                            speech_frame_count += 1
                
                if not frames:
                    continue

                speech_probability = speech_frame_count / len(frames)

                if is_recording:
                    recording_buffer.extend(frames)
                    if speech_probability > VAD_SPEECH_THRESHOLD:
                        silence_counter = 0
                    else:
                        silence_counter += len(frames)

                    if silence_counter > SILENCE_FRAMES_THRESHOLD or len(recording_buffer) > MAX_RECORDING_FRAMES:
                        print(f"[*] Dừng ghi âm. Tổng số khung: {len(recording_buffer)}")
                        final_waveform = torch.cat(recording_buffer, dim=1)
                        int16_tensor = (final_waveform.squeeze() * 32767).to(torch.int16)
                        audio_bytes = int16_tensor.cpu().numpy().tobytes()
                        transcript = await audio_processor.process_audio(audio_bytes)
                        print(transcript)
                        if transcript:
                            if manager.cancel_pipeline(client_id, llm_client.cleanup_fn):
                            # Thông báo client dừng audio nếu cần
                                await manager.send_json_to_client({"type": "control", "event": "interrupt"}, client_id)
                            
                            await manager.send_json_to_client({ 
                                "type": "transcript", 
                                "transcript": transcript
                            }, client_id)
                            # 🔥 TẠO pipeline mới
                            task = asyncio.create_task(
                                llm_tts_pipeline(client_id, transcript, llm_client, manager)
                            )
                            manager.set_pipeline_task(client_id, task, llm_client.cleanup_fn)

                        is_recording = False
                        recording_buffer.clear()
                        silence_counter = 0
                
                elif speech_probability > VAD_SPEECH_THRESHOLD:
                    print("[*] Bắt đầu ghi âm...")
                    is_recording = True
                    recording_buffer.extend(frames)
                    silence_counter = 0

            except Exception as e:
                raise e
    except WebSocketDisconnect:
        print("Client disconnected")
        await manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.disconnect(client_id)
    finally:
        print(f"Closing connection for client {client_id}.")
        await manager.disconnect(client_id)


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    client_audio_connections[client_id] = websocket
    print(f"🔊 New audio client: {client_id}")
    audio_buffer = bytearray()
    recording_buffer = []
    is_recording = False
    silence_counter = 0

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)

                if len(audio_buffer) < CHUNK_SIZE_BYTES:
                    continue

                chunk_bytes = audio_buffer[:CHUNK_SIZE_BYTES]
                del audio_buffer[:CHUNK_SIZE_BYTES]

                frames = []
                speech_frame_count = 0
                frame_byte_size = FRAME_SIZE * BYTES_PER_SAMPLE
                for i in range(0, len(chunk_bytes), frame_byte_size):
                    frame_data = chunk_bytes[i:i + frame_byte_size]
                    if len(frame_data) < frame_byte_size:
                        continue # Bỏ qua frame cuối nếu không đủ byte

                    # Gọi hàm xử lý mới, thuần túy
                    is_speech, waveform = process_frame(frame_data)
                    
                    if waveform is not None:
                        frames.append(waveform)
                        if is_speech:
                            speech_frame_count += 1
                
                if not frames:
                    continue

                speech_probability = speech_frame_count / len(frames)

                if is_recording:
                    recording_buffer.extend(frames)
                    if speech_probability > VAD_SPEECH_THRESHOLD:
                        silence_counter = 0
                    else:
                        silence_counter += len(frames)

                    if silence_counter > SILENCE_FRAMES_THRESHOLD or len(recording_buffer) > MAX_RECORDING_FRAMES:
                        print(f"[*] Dừng ghi âm. Tổng số khung: {len(recording_buffer)}")
                        final_waveform = torch.cat(recording_buffer, dim=1)
                        int16_tensor = (final_waveform.squeeze() * 32767).to(torch.int16)
                        audio_bytes = int16_tensor.cpu().numpy().tobytes()
                        transcript = await audio_processor.process_audio(audio_bytes)
                        print(transcript)
                        # TODO: send transcript to tts through socket
                        if client_id in client_audio_connections:
                            client_ws = client_audio_connections[client_id]
                            await client_ws.send_text(transcript)

                        is_recording = False
                        recording_buffer.clear()
                        silence_counter = 0
                
                elif speech_probability > VAD_SPEECH_THRESHOLD:
                    print("[*] Bắt đầu ghi âm...")
                    is_recording = True
                    recording_buffer.extend(frames)
                    silence_counter = 0
            except Exception as e:
                raise e
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        del client_audio_connections[client_id]
        print("WebSocket connection closed.")

# Reverse proxy các route
llm_app = reverse_http_app(base_url="http://localhost:3000/")

app.mount("/llm", llm_app)

tts_app = reverse_http_app(base_url="http://localhost:8002/")

app.mount("/tts", tts_app)

app.openapi = custom_openapi

if __name__ == "__main__":
    import socket
    import ssl

    import uvicorn

    def find_available_port(start_port=3000, max_port=5000):
        for port in range(start_port, max_port + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("0.0.0.0", port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No available ports found")
    available_port = find_available_port()
    print(f"Starting server on port {available_port}")
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile="ssl/cert.pem", keyfile="ssl/key.pem")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=1005,
        ssl_keyfile="ssl/key.pem",
        ssl_certfile="ssl/cert.pem",
    )
