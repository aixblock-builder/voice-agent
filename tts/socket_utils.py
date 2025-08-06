import asyncio
import websockets
import json
import ssl
from fastapi import FastAPI, WebSocket
from typing import Dict
import uuid
import json

LLM_WS_URL = "ws://localhost:8001/ws/stream-token"

ssl_context = ssl._create_unverified_context()
async def send_transcript_and_get_response(text: str) -> str:
    uri = LLM_WS_URL
    try:
        # Only for test
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            # Gửi transcript đến LLM
            await websocket.send(json.dumps({ "text": text }))

            response_text = ""
            while True:
                token = await websocket.recv()
                if token == "[DONE]":
                    break
                response_text += token
            return response_text

    except Exception as e:
        print(f"❌ WebSocket client error: {e}")
        return ""
    