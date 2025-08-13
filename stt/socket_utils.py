import asyncio
from fastapi import WebSocket
from typing import Dict
import uuid

class ConnectionManager:
    """Quản lý các kết nối WebSocket từ frontend."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.tts_tasks: Dict[str, asyncio.Task] = {}
        self.pipeline_tasks: dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self.active_connections[client_id] = websocket
        print(f"Client '{client_id}' đã kết nối. Tổng số client: {len(self.active_connections)}")
        return client_id

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections.pop(client_id)
            try:
                await websocket.close()
                print(f"Connection for {client_id} closed gracefully by server.")
            except RuntimeError as e:
                print(f"Could not send close frame to {client_id} (already disconnected by client): {e}")
        else:
            print(f"Warning: Attempted to disconnect non-existent client_id: {client_id}")

    async def send_to_client(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def send_json_to_client(self, message: Dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    def set_tts_task(self, client_id: str, task: asyncio.Task):
        # Hủy task cũ nếu còn
        if client_id in self.tts_tasks and not self.tts_tasks[client_id].done():
            self.tts_tasks[client_id].cancel()
        self.tts_tasks[client_id] = task

    def set_pipeline_task(self, client_id: str, task: asyncio.Task, cleanup_fn):
        # Huỷ pipeline cũ nếu còn chạy
        old = self.pipeline_tasks.get(client_id)
        if old and not old.done():
            old.cancel()
            cleanup_fn(client_id)
        self.pipeline_tasks[client_id] = task

    def cancel_pipeline(self, client_id: str, cleanup_fn):
        task = self.pipeline_tasks.get(client_id)
        if task and not task.done():
            task.cancel()
            cleanup_fn(client_id)
            return True
        return False

