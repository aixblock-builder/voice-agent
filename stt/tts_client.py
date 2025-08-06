import asyncio
import json
from typing import Callable, Optional
import websockets
from websockets.client import WebSocketClientProtocol
from app import manager

class TTSClient:
    """
    Quản lý kết nối lâu dài đến dịch vụ TTS, gửi văn bản và nhận lại
    luồng âm thanh.
    """
    """
    Phiên bản đơn giản: Chỉ chuyển tiếp tin nhắn JSON từ TTS về Gateway.
    """
    def __init__(self, uri: str, on_json_callback: Callable):
        self.uri = uri
        self.on_json_callback = on_json_callback # Callback để xử lý JSON
        self.websocket: Optional[WebSocketClientProtocol] = None

    async def connect_and_listen(self):
        while True:
            try:
                print("TTSClient: Đang kết nối tới TTS...")
                async with websockets.connect(self.uri, max_size=5 * 1024 * 1024) as ws:
                    self.websocket = ws
                    print("✅ TTSClient: Kết nối thành công tới TTS.")
                    
                    async for message in self.websocket:
                        # Nhận được tin nhắn JSON, gọi thẳng callback
                        await self.on_json_callback(message)

            except Exception as e:
                print(f"🔌 TTSClient: Mất kết nối hoặc lỗi: {e}. Thử lại sau 5 giây...")
                self.websocket = None
                await asyncio.sleep(5)

    async def request_audio_stream(self, client_id: str, text: str):
        """
        Gửi yêu cầu để TTS bắt đầu quá trình tạo và stream audio.
        """
        if self.websocket:
            payload = {"client_id": client_id, "text": text}
            await self.websocket.send(json.dumps(payload))
            print(f"TTSClient: Đã gửi yêu cầu cho client '{client_id}'.")
        else:
            print("TTSClient: Lỗi - Chưa kết nối tới TTS.")
    
    async def close(self):
        """
        Ngắt kết nối WebSocket và dừng tác vụ nền một cách an toàn.
        """
        print("TTSClient: Bắt đầu quá trình đóng...")
        self.is_running = False # Báo cho vòng lặp connect_and_listen dừng lại
        if self.websocket and not self.websocket.closed:
            await self.websocket.close(code=1000, reason='Client shutting down')
            print("TTSClient: Kết nối WebSocket đã được đóng.")

async def forward_tts_response_to_client(message_str: str):

    try:
        # Tách client_id từ chính JSON nhận được
        response_data = json.loads(message_str)
        client_id = response_data.get("client_id")
        response_data["type"] = "audio"

        if client_id:
            # Gửi toàn bộ object JSON này về client
            await manager.send_json_to_client(response_data, client_id)
        else:
            print("Gateway: Nhận được tin nhắn từ TTS nhưng thiếu client_id.")
            
    except json.JSONDecodeError:
        print(f"Gateway: Không thể parse JSON từ TTS: {message_str}")