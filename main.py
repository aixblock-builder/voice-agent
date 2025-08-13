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
from services_manager import (
    cancel_service,
    list_services,
    service_status,
    start_service,
)
from starlette.routing import Mount
from model import MyModel, mcp
from utils.chat_history import ChatHistoryManager
import json
from starlette.websockets import WebSocket
import subprocess
import atexit
from utils_voice_agent import (
    run_stt_app_func,
    run_tts_app_func,
    stop_stt_app,
    stop_tts_app,
    tts_proc,
    stt_proc,
    ensure_portaudio_installed,
)
from starlette.websockets import WebSocket

ensure_portaudio_installed()

subprocess.run("venv/bin/python load_model.py", shell=True)


# Models for request validation
class InstallServiceRequest(BaseModel):
    git: str


class ServiceInfoRequest(BaseModel):
    directory: str
    port_map: Optional[int] = None


class StopServiceRequest(BaseModel):
    port_map: int
    directory: Optional[str] = None


class DashboardRequest(BaseModel):
    directory: str


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

# Prometheus and Loki configuration
LOKI_URL = os.getenv("LOKI_URL", "http://207.246.109.178:3100")
JOB_NAME = os.getenv("JOB_NAME", "test-fastapi")
PUSH_GATEWAY_URL = os.getenv("PUSH_GATEWAY_URL", "http://207.246.109.178:9091")
JOB_INTERVAL = int(os.getenv("JOB_INTERVAL", 60))

model = MyModel()
chat_history = ChatHistoryManager(persist_directory="./chroma_db_history")


class ActionRequest(BaseModel):
    command: str
    params: Dict[str, Any]
    doc_file_urls: Optional[Union[str, List[str]]] = None
    session_id: Optional[str] = None
    use_history: Optional[bool] = True


@app.websocket("/ws/stream-token")
async def websocket_llm(websocket: WebSocket):
    await websocket.accept()
    print("Connected")
    try:
        while True:
            data = await websocket.receive_text()
            print(data)
            message = json.loads(data)
            response = model.action(
                "predict",
                **{"prompt": message["text"], "session_id": ""},
            )
            print(response["result"][0]["result"][0]["value"])

            await websocket.send_json(
                {
                    "client_id": message["client_id"],
                    "response": response["result"][0]["result"][0]["value"],
                }
            )
    except WebSocketDisconnect:
        print("client disconnected")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        await websocket.close()


@app.post("/init-voice-agent")
async def start_voice_agent(
    stt_config: Optional[Dict[str, Any]] = Body(None),
    tts_config: Optional[Dict[str, Any]] = Body(None),
):
    stt_id = await start_service(
        name="stt",
        config=stt_config,
        health_url="https://127.0.0.1:1005/health-check",
        run_fn_blocking=run_stt_app_func,
        stop_fn=stop_stt_app,
    )
    tts_id = await start_service(
        name="tts",
        config=tts_config,
        health_url="http://127.0.0.1:1006/health-check",
        run_fn_blocking=run_tts_app_func,
        stop_fn=stop_tts_app,
    )
    return {
        "message": "starting",
        "services": {
            "stt": stt_id,
            "tts": tts_id,
            "socket": {
                "port": 1005,
                "protocol": "wss",
                "endpoint": "/ws/audio",
            },
        },
    }


@app.get("/voice-agent/status")
async def status_all():
    return await list_services()


@app.get("/voice-agent/status/{service_id}")
async def status_one(service_id: str):
    data = await service_status(service_id)
    data["socket"] = {
        "port": 1005,
        "protocol": "wss",
        "endpoint": "/ws/audio",
    }
    return data


class CancelReq(BaseModel):
    port: int | None = None


@app.post("/voice-agent/cancel/{service_id}")
async def cancel_one(service_id: str, body: CancelReq):
    data = await cancel_service(service_id, kill_by_port=body.port)
    return data


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
    try:
        result = model.action(
            "mcp_register_payload",
            name=name,
            endpoint=endpoint,
            auth_token=auth_token,
            tools=tools,
            memoryConnection=memoryConnection,
            storageConnection=storageConnection,
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
    if stt_proc:
        stt_proc.terminate()
    if tts_proc:
        tts_proc.terminate()


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
