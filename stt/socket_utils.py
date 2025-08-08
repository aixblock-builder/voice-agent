import asyncio
from fastapi import WebSocket
from typing import Dict
import uuid

class ConnectionManager:
    """Quản lý các kết nối WebSocket từ frontend."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.tts_tasks: Dict[str, asyncio.Task] = {}
        self.tts_active_flags: Dict[str, bool] = {}

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

    def interrupt_tts_if_any(self, client_id: str) -> bool:
        """
        Ngắt nếu đang phát TTS hoặc vừa mới phát xong (nhưng client có thể chưa phát hết).
        """
        has_interrupted = False

        # Gửi interrupt nếu flag đang phát TTS là True
        if self.tts_active_flags.get(client_id, False):
            print(f"[TTS] Ngắt TTS đang hoạt động cho client {client_id}")
            self.tts_active_flags[client_id] = False  # reset cờ
            has_interrupted = True

        # Nếu task còn chạy thì cancel luôn
        if client_id in self.tts_tasks:
            task = self.tts_tasks[client_id]
            if not task.done():
                print(f"[TTS] Cancel task TTS đang chạy cho client {client_id}")
                task.cancel()
                has_interrupted = True

        return has_interrupted
