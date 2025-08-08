import asyncio
import json
import ssl
from typing import Callable, Dict, Optional
from socket_utils import ConnectionManager
import websockets
from websockets.client import WebSocketClientProtocol
import httpx

TTS_HTTP_URL = "http://localhost:8002/tts-only"

class LLMClient:
    """
    Quản lý kết nối lâu dài đến LLM, xử lý stream token và điều phối
    các phản hồi.
    """
    def __init__(self, uri: str, manager: ConnectionManager, on_response_callback: Callable):
        """
        Khởi tạo client.
        
        Args:
            uri (str): Địa chỉ WebSocket của LLM.
            on_response_callback (Callable): Một hàm async sẽ được gọi khi
                                            nhận được phản hồi đầy đủ từ LLM.
                                            Hàm này cần nhận (client_id, response_text).
        """
        self.uri = uri
        self.on_response_callback = on_response_callback
        self.manager = manager
        self.websocket: Optional[WebSocketClientProtocol] = None
        # Dùng để ghép các token cho mỗi yêu cầu
        self.response_buffers: Dict[str, str] = {}
        # Cấu hình SSL (chỉ cho môi trường test)
        self.is_running = False
        self.pending_futures: Dict[str, asyncio.Future] = {}

    async def connect_and_listen(self):
        """
        Tác vụ nền: Kết nối, lắng nghe vĩnh viễn và tự động kết nối lại.
        """
        self.is_running = True
        while self.is_running:
            try:
                print("LLMClient: Đang kết nối tới LLM...")
                async with websockets.connect(self.uri) as ws:
                    self.websocket = ws
                    print("✅ LLMClient: Kết nối thành công tới LLM.")
                    
                    # Vòng lặp lắng nghe tin nhắn từ LLM
                    async for message_str in self.websocket:
                        await self._handle_incoming_message(message_str)

            except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
                print(f"🔌 LLMClient: Mất kết nối tới LLM. Lỗi: {e}. Thử lại sau 5 giây...")
            except Exception as e:
                print(f"❌ LLMClient: Lỗi không xác định: {e}. Thử lại sau 5 giây...")
            if not self.is_running:
                break
            
            self.websocket = None
            await asyncio.sleep(5)
        print("LLMClient: Đã dừng tác vụ nền.")

    async def _handle_incoming_message(self, message: str):
        """
        Xử lý từng tin nhắn nhận được từ LLM, ghép token và gọi callback khi hoàn tất.
        """
        try:
            # Giả định LLM trả về JSON chứa client_id và token
            data = json.loads(message)
            client_id = data.get("client_id")
            token = data.get("token")

            if not client_id or token is None:
                return

            # Nếu không ai chờ kết quả của client_id này → bỏ qua
            if client_id not in self.pending_futures:
                return

            buf = self.response_buffers.setdefault(client_id, "")

            # Nếu là token kết thúc
            if token == "[DONE]":
                full = self.response_buffers.pop(client_id, "")
                print(f"LLMClient: Hoàn tất phản hồi cho client '{client_id}'.")
                # ► Trả kết quả cho pipeline nếu có Future đang chờ
                fut = self.pending_futures.get(client_id)
                if fut and not fut.done():
                    fut.set_result(full)
                else:
                    # fallback: gọi callback cũ (nếu bạn vẫn muốn hỗ trợ đường cũ)
                    await self.on_response_callback(client_id, full, self.manager)
            # Nếu là token bình thường
            else:
                self.response_buffers[client_id] = buf + token

        except json.JSONDecodeError:
            print(f"LLMClient: Lỗi - Nhận được tin nhắn không phải JSON: {message}")
        except Exception as e:
            print(f"LLMClient: Lỗi khi xử lý tin nhắn: {e}")

    def cleanup_fn(self, client_id):
        fut = self.pending_futures.pop(client_id, None)
        if fut and not fut.done():
            fut.cancel()
        self.response_buffers.pop(client_id, None)

    async def request_response(self, client_id: str, text: str):
        """
        Gửi một yêu cầu mới đến LLM qua kết nối đang có.
        """
        if not self.websocket:
            print("LLMClient: Lỗi - Không thể gửi yêu cầu, chưa kết nối tới LLM.")
            return
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.pending_futures[client_id] = fut
        payload = {
            "client_id": client_id,
            "text": text
        }
        await self.websocket.send(json.dumps(payload))
        print(f"LLMClient: Đã gửi yêu cầu cho client '{client_id}'.")
        try:
            return await fut
        except Exception as e:
            print(f"LLMClient: Lỗi khi gửi yêu cầu: {e}")
        finally:
            self.pending_futures.pop(client_id, None)

    async def close(self):
        """
        Ngắt kết nối WebSocket và dừng tác vụ nền một cách an toàn.
        """
        print("LLMClient: Bắt đầu quá trình đóng...")
        self.is_running = False # Báo cho vòng lặp connect_and_listen dừng lại
        if self.websocket:
            await self.websocket.close(code=1000, reason='Client shutting down')
            print("LLMClient: Kết nối WebSocket đã được đóng.")

async def llm_tts_pipeline(client_id, prompt, llm_client, manager):
    try:
        response_text = await llm_client.request_response(client_id, prompt)
        if not response_text:
            await manager.send_json_to_client({"error": "LLM returned empty."}, client_id)
            return
        await get_audio_from_tts_service(client_id, response_text, manager)
    except asyncio.CancelledError:
            # Task bị huỷ khi user gửi prompt mới → im lặng thoát
            print(f"[PIPE] Pipeline client {client_id} bị huỷ (prompt mới).")
    except Exception as e:
        print(f"[PIPE] Lỗi pipeline: {e}")
        await manager.send_json_to_client({"error": "Internal pipeline error"}, client_id)

async def handle_llm_response(client_id: str, response_text: str, manager: ConnectionManager):
    """
    Hàm này được gọi khi có phản hồi đầy đủ từ LLM.
    Nó sẽ kích hoạt bước tiếp theo: yêu cầu TTS tạo giọng nói.
    """
    print(f"Gateway: Nhận text từ LLM cho client '{client_id}', gửi qua TTS...")
    task = asyncio.create_task(get_audio_from_tts_service(client_id, response_text, manager))
    manager.set_tts_task(client_id, task)

async def get_audio_from_tts_service(client_id: str, text: str, manager: ConnectionManager):
    """
    Hàm này gọi đến TTS service qua HTTP và stream phản hồi về cho client.
    """
    params = {
        "client_id": client_id,
        "text": text,
        "speaker_wav": "example_female.wav",
        "language": "en"
    }
    
    try:
        # Sử dụng httpx.AsyncClient để thực hiện streaming request
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", TTS_HTTP_URL, params=params) as response:
                print(f"Gateway: Bắt đầu nhận stream từ TTS cho client '{client_id}'")
                
                # Đọc từng dòng trong stream (mỗi dòng là một JSON)
                async for line in response.aiter_lines():
                    if line:
                        try:
                            # Parse JSON và gửi về frontend qua WebSocket
                            json_data = json.loads(line)
                            json_data["type"] = "audio"
                            await manager.send_json_to_client(json_data, client_id)
                        except json.JSONDecodeError:
                            print(f"Gateway: Lỗi khi parse JSON từ TTS: {line}")
                            # Gửi tin nhắn lỗi về cho client
                            await manager.send_json_to_client({"error": "Invalid JSON from TTS."}, client_id)
    
    except asyncio.CancelledError:
        print(f"[TTS] Stream bị hủy giữa chừng cho client {client_id}")
        raise
                            
    except httpx.RequestError as e:
        print(f"Gateway: Lỗi khi gọi đến TTS service: {e}")
        # Gửi tin nhắn lỗi về cho client
        await manager.send_json_to_client({"error": "TTS service is unavailable."}, client_id)