import asyncio, json, re
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse

# pip install "realtimetts[kokoro]" fastapi uvicorn
from factory import TTSFactory

# kokoro config
config = {
    "engine_name": "kokoro",
    "engine_kwargs": {
        "voice": "af_heart",
    }
}
# xtts config
# config = {
#     "engine_name": "coqui",
#     "engine_kwargs": {
#         "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
#         "specific_model": "v2.0.2",
#         "voices_path": "./speakers",
#     }
# }
# plugin = TTSFactory.create("coqui", **config["engine_kwargs"])
engine = None
TextToAudioStream = None
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, TextToAudioStream, config
    from RealtimeTTS import TextToAudioStream as _TextToAudioStream
    TextToAudioStream = _TextToAudioStream
    try:
        TTSFactory._ensure_plugins_loaded()
    except Exception:
        pass
    print("[Startup] All preloads done.")
    plugin = TTSFactory.create(config["engine_name"], **config["engine_kwargs"])
    engine = plugin.get_engine() 
    print("[Startup] Engine ready.")
    try:
        yield
    finally:
        print("[Shutdown] Engines cache cleared.")

app = FastAPI(
    title="TTS model",
    lifespan=lifespan,
    openapi_url="/swagger.json",
    docs_url="/swagger",
)

def split_sentences(text: str) -> List[str]:
    # tách câu đơn giản; tùy ý thay thế cho phù hợp tiếng Việt
    parts = re.split(r'(?<=[\.\?\!…。！？])\s+', text.strip())
    return [p for p in parts if p]

# ====== STREAMING HANDLER ======
async def synth_sentence_pcm(sentence: str) -> bytes:
    q: asyncio.Queue[bytes] = asyncio.Queue()
    done = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_chunk(b: bytes):
        loop.call_soon_threadsafe(q.put_nowait, b)

    def on_stop():
        loop.call_soon_threadsafe(done.set)

    stream = TextToAudioStream(
        engine=engine,
        muted=True,
        on_audio_stream_stop=on_stop,
    )
    stream.play_async(
        buffer_threshold_seconds=0.0,
        on_audio_chunk=on_chunk,
    )
    stream.feed(sentence)

    chunks: List[bytes] = []
    while True:
        if done.is_set() and q.empty():
            break
        try:
            chunk = await asyncio.wait_for(q.get(), timeout=0.2)
            chunks.append(chunk)
        except asyncio.TimeoutError:
            continue
    return b"".join(chunks)

@app.get("/")
async def root():
    return FileResponse("index.html", media_type="text/html")

# ====== API ======
@app.get("/tts-only")
async def tts(
    text: str = Query(..., min_length=1, description="Nội dung cần đọc"),
):
    sentences = split_sentences(text)

    async def generator() -> AsyncGenerator[bytes, None]:
        for index, sentence in enumerate(sentences):
            audio_bytes = await synth_sentence_pcm(sentence)

            # CẢNH BÁO: list(audio_bytes) rất to; cân nhắc dùng base64 (xem ghi chú bên dưới)
            json_data = {
                "text_index": index,
                "is_last": index == len(sentences) - 1,
                "text_length": len(sentence),
                "text": [sentence],
                "audio": list(audio_bytes),  # hoặc "audio_b64": base64.b64encode(audio_bytes).decode()
                "sample_rate": 24000,
                "sample_format": "s16le",
                "channels": 1,
            }
            line = json.dumps(json_data, ensure_ascii=False) + "\n"
            yield line.encode("utf-8")

    return StreamingResponse(generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=1006,
        # ssl_keyfile="ssl/key.pem",
        # ssl_certfile="ssl/cert.pem",
    )
