import asyncio
import os
from typing import Any, Dict, Optional, List, Union, AsyncIterator
import requests
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Body,
    WebSocketDisconnect,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel

from starlette.routing import Mount
from model import MyModel, mcp
from speech_to_text.plugin_loader import *
from text_to_speech.plugin_loader import *
from speech_to_text.asr_base import create_plugin as create_asr_plugin
from text_to_speech.tts_base import create_plugin as create_tts_plugin
from utils.chat_history import ChatHistoryManager
import json
from starlette.websockets import WebSocket
import atexit
import base64
from starlette.websockets import WebSocket

# Thêm vào đầu file, sau các import
import threading
from concurrent.futures import ThreadPoolExecutor

# Global TTS instance management
_global_tts_instances = {}
_instance_lock = threading.Lock()
_tts_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts_pool")

app = FastAPI(
    title="My model",
    openapi_url="/swagger.json",
    docs_url="/swagger",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = MyModel()
chat_history = ChatHistoryManager(persist_directory="./chroma_db_history")

class ActionRequest(BaseModel):
    command: str
    params: Dict[str, Any]
    doc_file_urls: Optional[Union[str, List[str]]] = None
    session_id: Optional[str] = None
    use_history: Optional[bool] = True

active_plugins_asr: Dict[str, Any] = {}
active_plugins_tts: Dict[str, Any] = {}
websocket_connections: Dict[str, WebSocket] = {}
agent_connections: Dict[str, Dict[str, Any]] = {}

class ASRConfig(BaseModel):
    plugin_type: str  # "whisper", "huggingface", etc
    config_model: Dict[str, Any]  # model-specific config

class TTSConfig(BaseModel):
    plugin_type: str  # "xtts2", "kokoro", etc
    config_model: Dict[str, Any]  # model-specific config

class InitAgentRequest(BaseModel):
    name: str
    agent_name: str
    endpoint: str
    auth_token: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    memoryConnection: Optional[Dict[str, Any]] = None
    storageConnection: Optional[Dict[str, Any]] = None
    asr_config: Optional[ASRConfig] = None
    tts_config: Optional[TTSConfig] = None

class ConversationState:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.is_initialized = False
        self.audio_buffer = []
        self.is_processing = False
        self.conversation_id = f"conv_{agent_name}_{asyncio.get_event_loop().time()}"

@app.websocket("/conversation")
async def websocket_conversation_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    conversation_state = None
    
    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "conversation_initiation_client_data":
                conversation_state = await handle_init_conversation(websocket, data)
            elif message_type == "user_audio_chunk":
                if conversation_state:
                    await handle_audio_chunk(websocket, data, conversation_state)
            elif message_type == "pong":
                await handle_pong(data)
                
    except WebSocketDisconnect:
        print(f"[Agent] Client disconnected")
    except Exception as e:
        print(f"[Agent] Error: {e}")
        await websocket.close()


@app.post("/mcp/register")
async def register_mcp_server(
    name: str = Body(...),
    endpoint: str = Body(...),
    auth_token: str = Body(None),
    tools: List[Dict[str, Any]] = Body(None),
    memoryConnection: Dict[str, Any] = Body(None),
    storageConnection: Dict[str, Any] = Body(None),
):
    """Register a remote MCP server"""
async def handle_init_conversation(websocket: WebSocket, data: Dict[str, Any]) -> ConversationState:
    """Handle conversation initialization"""
    try:
        # Extract config
        config = data.get("conversation_config_override", {})
        agent_name = data.get("agent_name", "default_agent")
        agent_config = config.get("agent", {})
        prompt = agent_config.get("prompt", {}).get("prompt", "You are a helpful assistant")
        first_message = agent_config.get("first_message", "Hello! How can I help you today?")
        
        # Create conversation state
        conversation_state = ConversationState(agent_name)
        
        # Send ready signal
        await websocket.send_text(json.dumps({
            "type": "conversation_initiation_metadata",
            "conversation_id": conversation_state.conversation_id
        }))
        
        conversation_state.is_initialized = True
        
        # Send first message if provided
        if first_message:
            # Generate TTS for first message
            first_audio = await text_to_speech(first_message, agent_name)
            
            if first_audio:
                await websocket.send_text(json.dumps({
                    "type": "audio",
                    "audio_event": {
                        "audio_base_64": first_audio
                    }
                }))
            
            # Send agent response text
            await websocket.send_text(json.dumps({
                "type": "agent_response",
                "agent_response_event": {
                    "agent_response": first_message
                }
            }))
        
        return conversation_state
        
    except Exception as e:
        print(f"[Agent] Init error: {e}")
        raise

async def handle_audio_chunk(websocket: WebSocket, data: Dict[str, Any], state: ConversationState):
    """Handle incoming audio chunk"""
    if not state.is_initialized or state.is_processing:
        # Buffer audio if not ready
        state.audio_buffer.append(data.get("user_audio_chunk"))
        return
    
    state.is_processing = True
    
    try:
        # Get audio data
        audio_base64 = data.get("user_audio_chunk")
        if not audio_base64:
            return
            
        # Decode audio
        audio_data = base64.b64decode(audio_base64)
        
        # Speech to Text using active plugin
        transcript = await speech_to_text_with_plugin(audio_data, state.agent_name)
        
        if transcript:
            # Send user transcript
            await websocket.send_text(json.dumps({
                "type": "user_transcript", 
                "user_transcription_event": {
                    "user_transcript": transcript
                }
            }))

            await asyncio.sleep(0.05)
            
            # # Generate AI response
            # # Generate AI response
            # ai_response = await generate_ai_response(transcript, state.agent_name, state.conversation_id)
            ai_response = "It's interesting you mentioned the queen and the sister pair—sounds like a mystery! If you're Jessica and you've been watching through a cold one, maybe you're part of a larger story or a group with a secret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigation?"
            print("====ai_response====", ai_response)
            
            # Send agent response text
            await websocket.send_text(json.dumps({
                "type": "agent_response",
                "agent_response_event": {
                    "agent_response": str(ai_response)
                }
            }))

            await asyncio.sleep(0.05)
            
            # # Generate and send audio response
            chunk_size = 80  # Điều chỉnh theo nhu cầu
            chunks = [ai_response[i:i+chunk_size] for i in range(0, len(ai_response), chunk_size)]

            for chunk in chunks:
                print("====chunk====", chunk)
                audio_response = await text_to_speech(chunk, state.agent_name)
                if audio_response:
                    await websocket.send_text(json.dumps({
                        "type": "audio",
                        "audio_event": {
                            "audio_base_64": audio_response
                        }
                    }))
                    await asyncio.sleep(0.1)
        
    except Exception as e:
        print(f"[Agent] Audio processing error: {e}")
    finally:
        state.is_processing = False

async def handle_pong(data: Dict[str, Any]):
    """Handle pong message"""
    event_id = data.get("event_id")
    print(f"[Agent] Received pong for event: {event_id}")

async def speech_to_text_with_plugin(audio_data: bytes, agent_name: str) -> Optional[str]:
    """Convert speech to text using active ASR plugin"""
    try:
        # Get ASR plugin for this agent
        asr_plugin = active_plugins_asr.get(agent_name, active_plugins_asr.get(agent_name))
        print(active_plugins_asr)
        
        if not asr_plugin:
            print(f"[STT] No ASR plugin found for agent: {agent_name}")
            return None
        
        # Convert audio data to format expected by plugin
        # audio_file = convert_audio_to_wav(audio_data)
        
        # Run STT
        result = await asyncio.to_thread(asr_plugin.predict, audio_data)
        
        if isinstance(result, dict):
            return result.get('text', '')
        return str(result) if result else None
        
    except Exception as e:
        print(f"[STT] Error: {e}")
        return None

async def generate_ai_response(text: str, agent_name: str, session_id: str) -> str:
    """Generate AI response using model"""
    try:
        # Use existing model to generate response
        action_request = ActionRequest(
            command="predict",
            params={
                "prompt": text,
                "enable_function_calling": False,
            },
            session_id=session_id,
            use_history=True
        )
        
        response = model.action(
            action_request.command,
            **action_request.params,
            session_id=session_id,
            use_history=action_request.use_history
        )
        
        # Extract text from response
        if isinstance(response, dict):
            return response.get("response") or response.get("text") or "I'm sorry, I couldn't process that."
        else:
            return str(response) if response else "I'm sorry, I couldn't process that."
        
    except Exception as e:
        print(f"[AI] Response generation error: {e}")
        return "I'm sorry, there was an error processing your request."

def get_tts_instance(agent_name: str):
    """Lấy TTS instance từ global storage"""
    return _global_tts_instances.get(agent_name)

def set_tts_instance(agent_name: str, tts_instance):
    """Set TTS instance vào global storage với thread safety"""
    with _instance_lock:
        _global_tts_instances[agent_name] = tts_instance
        print(f"[TTS] Global instance set for agent: {agent_name}")

async def text_to_speech(text: str, agent_name: str) -> Optional[str]:
    """Convert text to speech using pre-loaded instance"""
    def _synthesize():
        # Lấy instance từ global storage trước
        tts_plugin = get_tts_instance(agent_name)
        
        # Fallback về active_plugins_tts nếu không có trong global
        if not tts_plugin:
            tts_plugin = active_plugins_tts.get(agent_name)
            if tts_plugin:
                # Lưu vào global cho lần sau
                set_tts_instance(agent_name, tts_plugin)
        
        if not tts_plugin:
            print(f"[TTS] No TTS plugin found for agent: {agent_name}")
            return None
        
        # Synthesize (model đã load sẵn)
        audio = tts_plugin.synthesize(text)
        buffer = tts_plugin.audio_to_bytes(audio)
        return base64.b64encode(buffer).decode("ascii")
    
    try:
        loop = asyncio.get_event_loop()
        audio_b64 = await loop.run_in_executor(_tts_thread_pool, _synthesize)
        return audio_b64
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return None

def convert_audio_to_wav(audio_data: bytes) -> str:
    """Convert audio data to WAV file for ASR plugin"""
    try:
        # Create temporary WAV file
        temp_file = f"/tmp/audio_{asyncio.get_event_loop().time()}.wav"
        
        # Assume audio_data is already in WAV format (16kHz, mono)
        # If not, you may need additional conversion
        with open(temp_file, 'wb') as f:
            f.write(audio_data)
            
        return temp_file
        
    except Exception as e:
        print(f"[Audio] Conversion error: {e}")
        return None

# Ping/Pong keep-alive
async def send_ping_to_clients():
    """Send periodic ping to all connected clients"""
    while True:
        try:
            await asyncio.sleep(30)  # Every 30 seconds
            
            for connection_id, connection_info in agent_connections.items():
                websocket = connection_info.get("websocket")
                if websocket:
                    event_id = f"ping_{asyncio.get_event_loop().time()}"
                    await websocket.send_text(json.dumps({
                        "type": "ping",
                        "ping_event": {
                            "event_id": event_id
                        }
                    }))
                    
        except Exception as e:
            print(f"[Ping] Error: {e}")

# Start ping task on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(send_ping_to_clients())

@app.post("/init_agent")
async def init_agent(request: InitAgentRequest):
    """Initialize agent with ASR plugin"""
    try:
        # Original MCP logic
        result = model.action(
            "mcp_register_payload",
            name=request.name,
            endpoint=request.endpoint,
            auth_token=request.auth_token,
            tools=request.tools,
            memoryConnection=request.memoryConnection,
            storageConnection=request.storageConnection,
        )
        
        # Initialize ASR plugin if provided
        if request.asr_config:
            asr_plugin = create_asr_plugin(
                request.asr_config.plugin_type,
                **request.asr_config.config_model
            ).load()
            active_plugins_asr[request.agent_name] = asr_plugin
        
        if request.tts_config:
            tts_plugin = create_tts_plugin(
                request.tts_config.plugin_type,
                **request.tts_config.config_model
            ).load()
            active_plugins_tts[request.agent_name] = tts_plugin
            set_tts_instance(request.agent_name, tts_plugin)  # Lưu vào global
            
            print(f"[TTS] TTS loaded and ready for agent: {request.agent_name}")
        
        return {"success": True, "result": result, "asr_enabled": bool(request.asr_config)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools"""
    try:
        result = model.action("mcp_list_tools")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/template_list")
async def get_prompt_template():
    """Get all prompt templates"""
    try:
        result = model.action("template_list")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/template_get")
async def get_prompt_template(template_name: str):
    """Get a prompt template"""
    try:
        result = model.action("template_get", template_name=template_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/template_delete")
async def delete_prompt_template(template_name: str):
    """Delete a prompt template"""
    try:
        result = model.action("template_delete", template_name=template_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/template_update")
async def update_prompt_template(
    template_name: str = Body(...),
    template_text: str = Body(...),
    input_variables: List[str] = Body(...),
    description: str = Body(...),
):
    """Update a prompt template"""
    try:
        result = model.action(
            "template_update",
            template_name=template_name,
            template_text=template_text,
            input_variables=input_variables,
            description=description,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools"""
    try:
        result = model.action("mcp_list_tools")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/action")
async def action(request: ActionRequest = Body(...)):
    try:
        parsed_params = request.params

        # Handle session management
        session_id = request.session_id
        print(f"session_id: {session_id}")
        if not session_id:
            session_result = chat_history.create_new_session()
            session_id = session_result["session_id"]
            print(
                f"🆕 Created new session: {session_id} with title: {session_result['title']}"
            )

        # Get conversation history if enabled and for predict command
        conversation_history = []
        if request.use_history and request.command.lower() == "predict":
            conversation_history = chat_history.get_session_history(session_id, limit=5)
            print(
                f"📚 Retrieved {len(conversation_history)} history turns for session {session_id}"
            )

        # Normalize URL list
        doc_file_urls = request.doc_file_urls
        if isinstance(doc_file_urls, str):
            doc_file_urls = [doc_file_urls]

        if doc_file_urls:
            # Convert URLs to list of temp file paths
            file_paths = fetch_file_paths_from_urls_sync(doc_file_urls)
            parsed_params["doc_files"] = file_paths
            parsed_params["docchat"] = True

        # Add history context for predict commands
        if request.command.lower() == "predict" and conversation_history:
            parsed_params["conversation_history"] = conversation_history
            parsed_params["session_id"] = session_id

        result = model.action(request.command, **parsed_params)

        # Save conversation to history if it's a predict command
        if request.command.lower() == "predict" and "result" in result:
            user_prompt = parsed_params.get("prompt", parsed_params.get("text", ""))
            bot_response = ""

            # Extract bot response from result
            if isinstance(result.get("result"), list) and len(result["result"]) > 0:
                first_result = result["result"][0]
                if "result" in first_result and len(first_result["result"]) > 0:
                    value = first_result["result"][0].get("value", {})
                    if "text" in value and isinstance(value["text"], list):
                        bot_response = value["text"][0] if value["text"] else ""

            print(bot_response)
            # if user_prompt and bot_response:
            #     doc_files_used = parsed_params.get("doc_files", [])
            #     chat_history.save_conversation_turn(
            #         session_id=session_id,
            #         user_message=user_prompt,
            #         bot_response=bot_response,
            #         doc_files=doc_files_used,
            #         metadata={"command": request.command}
            #     )

        # Add session_id to response
        result["session_id"] = session_id
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def fetch_file_paths_from_urls_sync(
    urls: List[str], save_dir: str = "downloads"
) -> List[str]:
    file_paths = []

    # Tạo thư mục nếu chưa tồn tại
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            filename = url.split("/")[-1] or "file"
            suffix = os.path.splitext(filename)[-1] or ".pdf"

            # Đảm bảo tên file không trùng lặp
            save_path = Path(save_dir) / filename
            counter = 1
            while save_path.exists():
                save_path = Path(save_dir) / f"{Path(filename).stem}_{counter}{suffix}"
                counter += 1

            # Ghi nội dung vào file
            with open(save_path, "wb") as f:
                f.write(response.content)

            file_paths.append(str(save_path))

        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            continue

    return file_paths


# V2 Collections API Endpoints
@app.post("/v2/collections")
async def create_new_collection(title: Optional[str] = None):
    """Create a new chat collection (session)"""
    try:
        session_result = chat_history.create_new_session(title)
        return {
            "id": session_result["session_id"],
            "title": session_result["title"],
            "message": "New collection created",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v2/collections")
async def get_all_collections(limit: int = 50):
    """Get list of all chat collections (sessions) with metadata"""
    try:
        sessions = chat_history.get_all_sessions(limit)
        # Convert sessions to collections format
        collections = []
        for session in sessions:
            collections.append(
                {
                    "id": session["session_id"],
                    "title": session.get("title", "Untitled Collection"),
                    "turn_count": session["turn_count"],
                    "first_message": session["first_message"],
                    "last_message": session["last_message"],
                    "created_at": session["created_at"],
                    "updated_at": session["updated_at"],
                    "doc_files_used": list(session["doc_files_used"]),
                }
            )

        return {"collections": collections, "count": len(collections), "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v2/collections/{collection_id}")
async def get_collection_history(collection_id: str, limit: int = 10):
    """Get conversation history for a collection (session)"""
    try:
        history = chat_history.get_session_history(collection_id, limit)
        return {"id": collection_id, "history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v2/collections/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete a chat collection (session) and all its history"""
    try:
        success = chat_history.delete_session(collection_id)
        if success:
            return {"message": f"Collection {collection_id} deleted successfully"}
        else:
            return {"message": f"Collection {collection_id} not found or already empty"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v2/collections/search")
async def search_collections(
    query: str, collection_id: Optional[str] = None, n_results: int = 5
):
    """Search for similar conversations across collections"""
    try:
        results = chat_history.search_similar_conversations(
            query, collection_id, n_results
        )
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def model_endpoint(data: Optional[Dict[str, Any]] = None):
    try:
        result = model.model(**(data or {}))
        if "share_url" in result:
            return RedirectResponse(url=result["share_url"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model")
async def model_endpoint(data: Optional[Dict[str, Any]] = None):
    try:
        result = model.model(**(data or {}))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model-trial")
async def model_trial(project: str, data: Optional[Dict[str, Any]] = None):
    try:
        result = model.model_trial(project, **(data or {}))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download")
async def download(project: str, data: Optional[Dict[str, Any]] = None):
    try:
        result = model.download(project, **(data or {}))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/downloads")
async def download_file(path: str):
    if not path:
        raise HTTPException(status_code=400, detail="File name is required")

    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, path)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(full_path, filename=os.path.basename(full_path))


sse = SseServerTransport("/messages/")

app.router.routes.append(Mount("/messages", app=sse.handle_post_message))


# Add documentation for the /messages endpoint
@app.get("/messages", tags=["MCP"], include_in_schema=True)
def messages_docs():
    pass


@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request):
    """
    SSE endpoint that connects to the MCP server

    This endpoint establishes a Server-Sent Events connection with the client
    and forwards communication to the Model Context Protocol server.
    """
    # Use sse.connect_sse to establish an SSE connection with the MCP server
    async with sse.connect_sse(request.scope, request.receive, request._send) as (
        read_stream,
        write_stream,
    ):
        # Run the MCP server with the established streams
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )


def cleanup():
    print("Stopping child apps...")


atexit.register(cleanup)

if __name__ == "__main__":
    import socket
    import ssl
    import sys
    import subprocess

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

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3000,
        # Bạn cũng có thể thêm các cấu hình khác ở đây
        # ssl_keyfile="ssl/key.pem",
        # ssl_certfile="ssl/cert.pem",
    )
