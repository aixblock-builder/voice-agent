import asyncio
import json
import base64
from typing import Dict, Any
from starlette.websockets import WebSocket
from handlers.stt_handler import speech_to_text_with_plugin
from handlers.tts_handler import text_to_speech
from handlers.llm_handler import generate_ai_response

# Global connections storage
websocket_connections: Dict[str, WebSocket] = {}
agent_connections: Dict[str, Dict[str, Any]] = {}

class ConversationState:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.is_initialized = False
        self.audio_buffer = []
        self.is_processing = False
        self.conversation_id = f"conv_{agent_name}_{asyncio.get_event_loop().time()}"

async def handle_init_conversation(
    websocket: WebSocket, data: Dict[str, Any]
) -> ConversationState:
    """Handle conversation initialization"""
    try:
        # Extract config
        config = data.get("conversation_config_override", {})
        agent_name = data.get("agent_name", "default_agent")
        agent_config = config.get("agent", {})
        prompt = agent_config.get("prompt", {}).get(
            "prompt", "You are a helpful assistant"
        )
        first_message = agent_config.get(
            "first_message", "Hello! How can I help you today?"
        )

        # Create conversation state
        conversation_state = ConversationState(agent_name)

        # Send ready signal
        await websocket.send_text(
            json.dumps(
                {
                    "type": "conversation_initiation_metadata",
                    "conversation_id": conversation_state.conversation_id,
                }
            )
        )

        conversation_state.is_initialized = True

        # Send first message if provided
        if first_message:
            # Generate TTS for first message
            first_audio = await text_to_speech(first_message, agent_name)
            
            if first_audio:
                await websocket.send_text(
                    json.dumps(
                        {"type": "audio", "audio_event": {"audio_base_64": first_audio}}
                    )
                )

            # Send agent response text
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "agent_response",
                        "agent_response_event": {"agent_response": first_message},
                    }
                )
            )

        return conversation_state

    except Exception as e:
        print(f"[Agent] Init error: {e}")
        raise

async def handle_audio_chunk(
    websocket: WebSocket, data: Dict[str, Any], state: ConversationState, model_instance
):
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
            print("====transcript====", transcript)
            await websocket.send_text(json.dumps({
                "type": "user_transcript", 
                "user_transcription_event": {
                    "user_transcript": transcript
                }
            }))

            await asyncio.sleep(0.05)

            ai_response = "It's interesting you mentioned the queen and the sister pair—sounds like a mystery! If you're Jessica and you've been watching through a cold one, maybe you're part of a larger story or a group with a secret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigation?"
            
            # Generate AI response
            # ai_response = await generate_ai_response(
            #     transcript, state.agent_name, state.conversation_id, model_instance
            # )
            print("====ai_response====", ai_response)
            
            # Send agent response text
            await websocket.send_text(json.dumps({
                "type": "agent_response",
                "agent_response_event": {
                    "agent_response": str(ai_response)
                }
            }))

            await asyncio.sleep(0.05)
            
            # Generate and send audio response
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
                    await websocket.send_text(
                        json.dumps(
                            {"type": "ping", "ping_event": {"event_id": event_id}}
                        )
                    )

        except Exception as e:
            print(f"[Ping] Error: {e}")

def get_websocket_connections():
    """Get all websocket connections"""
    return websocket_connections

def get_agent_connections():
    """Get all agent connections"""
    return agent_connections

def cleanup_connection(connection_id: str):
    """Clean up websocket connections"""
    try:
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]
        if connection_id in agent_connections:
            del agent_connections[connection_id]
        print(f"[WebSocket] Cleaned up connection: {connection_id}")
    except Exception as e:
        print(f"[WebSocket] Failed to cleanup connection {connection_id}: {e}")