import json
from fastapi import FastAPI, HTTPException, Request, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,StreamingResponse
from modeldownloader import check_stream2sentence_version,install_deepspeed_based_on_python_version
from tts_funcs import TTSWrapper,supported_languages,InvalidSettingsError
import os
from loguru import logger
from RealtimeTTS import TextToAudioStream, CoquiEngine
import asyncio

from socket_utils import send_transcript_and_get_response

DEVICE = os.getenv('DEVICE',"cuda")
OUTPUT_FOLDER = os.getenv('OUTPUT', 'output')
SPEAKER_FOLDER = os.getenv('SPEAKER', 'speakers')
MODEL_FOLDER = os.getenv('MODEL', 'models')
BASE_HOST = os.getenv('BASE_URL', '127.0.0.1:8020')
BASE_URL = os.getenv('BASE_URL', '127.0.0.1:8020')
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
MODEL_VERSION = os.getenv("MODEL_VERSION","v2.0.2")
LOWVRAM_MODE = os.getenv("LOWVRAM_MODE") == 'true'
DEEPSPEED = os.getenv("DEEPSPEED") == 'true'
USE_CACHE = os.getenv("USE_CACHE") == 'true'

# STREAMING VARS
STREAM_MODE = os.getenv("STREAM_MODE") == 'true'
STREAM_MODE_IMPROVE = os.getenv("STREAM_MODE_IMPROVE") == 'true'
STREAM_PLAY_SYNC = os.getenv("STREAM_PLAY_SYNC") == 'true'

if(DEEPSPEED):
  install_deepspeed_based_on_python_version()

app = FastAPI(
    title="TTS model",
    openapi_url="/swagger.json",
    docs_url="/swagger",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

XTTS = TTSWrapper(
    OUTPUT_FOLDER,SPEAKER_FOLDER,MODEL_FOLDER,LOWVRAM_MODE,MODEL_SOURCE,MODEL_VERSION,DEVICE,DEEPSPEED,USE_CACHE)

# Create version string
version_string = ""
if MODEL_SOURCE == "api" or MODEL_VERSION == "main":
    version_string = "lastest"
else:
    version_string = MODEL_VERSION

# Load model
if STREAM_MODE or STREAM_MODE_IMPROVE:
    # Load model for Streaming
    check_stream2sentence_version()

    logger.warning("'Streaming Mode' has certain limitations, you can read about them here https://github.com/daswer123/xtts-api-server#about-streaming-mode")

    if STREAM_MODE_IMPROVE:
        logger.info("You launched an improved version of streaming, this version features an improved tokenizer and more context when processing sentences, which can be good for complex languages like Chinese")
        
    model_path = XTTS.model_folder
    
    engine = CoquiEngine(specific_model=MODEL_VERSION,use_deepspeed=DEEPSPEED,local_models_path=str(model_path))
    stream = TextToAudioStream(engine)
else:
  logger.info(f"Model: '{version_string}' starts to load,wait until it loads")
  XTTS.load_model() 

if USE_CACHE:
    logger.info("You have enabled caching, this option enables caching of results, your results will be saved and if there is a repeat request, you will get a file instead of generation")

@app.get('/')
async def index():
    return FileResponse("index.html")

@app.get('/tts_stream')
async def tts_stream(request: Request, text: str = Query(), speaker_wav: str = Query(), language: str = Query()):
    # Validate local model source.
    if XTTS.model_source != "local":
        raise HTTPException(status_code=400,
                            detail="HTTP Streaming is only supported for local models.")
    # Validate language code against supported languages.
    if language.lower() not in supported_languages:
        raise HTTPException(status_code=400,
                            detail="Language code sent is either unsupported or misspelled.")
    
    try:
        llm_response = await send_transcript_and_get_response(text)
        print('llm_response',llm_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    async def generator():
        chunks = XTTS.process_tts_to_file(
            text=llm_response,
            speaker_name_or_path=speaker_wav,
            language=language.lower(),
            stream=True,
        )
        # Write file header to the output stream.
        yield XTTS.get_wav_header()
        async for chunk in chunks:
            # Check if the client is still connected.
            disconnected = await request.is_disconnected()
            if disconnected:
                break
            yield chunk

    return StreamingResponse(generator(), media_type='audio/x-wav')

@app.get('/tts_stream_new')
async def tts_stream(request: Request, text: str = Query(), speaker_wav: str = Query(), language: str = Query()):
    # Validate local model source.
    if XTTS.model_source != "local":
        raise HTTPException(status_code=400,
                            detail="HTTP Streaming is only supported for local models.")
    # Validate language code against supported languages.
    if language.lower() not in supported_languages:
        raise HTTPException(status_code=400,
                            detail="Language code sent is either unsupported or misspelled.")
    
    try:
        llm_response = await send_transcript_and_get_response(text)
        sentences = [s.strip() for s in llm_response.split(".") if s.strip()]
        print('sentences', sentences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    async def generator():
        for index, sentence in enumerate(sentences):
            if await request.is_disconnected():
                break

            audio_chunks = XTTS.process_tts_to_file(
                text=sentence,
                speaker_name_or_path=speaker_wav,
                language=language.lower(),
                stream=True,
            )

            audio_data = b"".join([chunk async for chunk in audio_chunks])
            json_data = {
                "text_index": index,
                "is_last": index == len(sentences) - 1,
                "text_length": len(sentence),
                "text": [sentence],
                "audio": list(audio_data)
            }

            yield json.dumps(json_data) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")

@app.get('/tts-only')
async def tts_only(request: Request, text: str = Query(), speaker_wav: str = Query(), language: str = Query()):
    # Validate local model source.
    if XTTS.model_source != "local":
        raise HTTPException(status_code=400,
                            detail="HTTP Streaming is only supported for local models.")
    # Validate language code against supported languages.
    if language.lower() not in supported_languages:
        raise HTTPException(status_code=400,
                            detail="Language code sent is either unsupported or misspelled.")
    
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    async def generator():
        for index, sentence in enumerate(sentences):
            if await request.is_disconnected():
                break

            audio_chunks = XTTS.process_tts_to_file(
                text=sentence,
                speaker_name_or_path=speaker_wav,
                language=language.lower(),
                stream=True,
            )

            audio_data = b"".join([chunk async for chunk in audio_chunks])
            json_data = {
                "text_index": index,
                "is_last": index == len(sentences) - 1,
                "text_length": len(sentence),
                "text": [sentence],
                "audio": list(audio_data)
            }

            yield json.dumps(json_data) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")

@app.websocket("/ws/generate-audio")
async def generate_audio(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected.")

    try:
        while True:
            data = await websocket.receive_json()
            client_id = data.get("client_id", 0)
            text = data.get("text")
            speaker_wav = data.get("speaker_wav", "example_female.wav")
            language = data.get("language", "en")

            if not all([text, speaker_wav, language]):
                await websocket.send_json({"error": "Missing required parameters."})
                continue

            if XTTS.model_source != "local":
                 await websocket.send_json({"error": "Streaming is only supported for local models."})
                 continue
            if language.lower() not in supported_languages:
                 await websocket.send_json({"error": "Language not supported."})
                 continue
            
            sentences = [s.strip() for s in text.split(".") if s.strip()]

            for index, sentence in enumerate(sentences):
                # Tạo audio chunks từ model TTS
                audio_chunks = XTTS.process_tts_to_file(
                    text=sentence,
                    speaker_name_or_path=speaker_wav,
                    language=language.lower(),
                    stream=True,
                )

                audio_data = b"".join([chunk async for chunk in audio_chunks])

                json_data = {
                    "text_index": index,
                    "is_last": index == len(sentences) - 1,
                    "text": sentence,
                    "audio": list(audio_data),
                    "client_id": client_id
                }

                await websocket.send_json(json_data)
                print(f"Sent chunk {index + 1}/{len(sentences)} for client.")

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason=f"Internal server error: {e}")

if __name__ == "__main__":
    import socket
    import ssl

    import uvicorn

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile="ssl/cert.pem", keyfile="ssl/key.pem")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=1006,
        # ssl_keyfile="ssl/key.pem",
        # ssl_certfile="ssl/cert.pem",
    )