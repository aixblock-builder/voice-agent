from fastapi import WebSocket
from typing import Dict
import uuid

class ConnectionManager:
    """Quản lý các kết nối WebSocket từ frontend."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

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
