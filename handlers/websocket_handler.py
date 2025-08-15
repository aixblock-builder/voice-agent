import asyncio
import json
import base64
import time
import webrtcvad
from typing import Dict, Any, Optional
from starlette.websockets import WebSocket
from handlers.stt_handler import speech_to_text_with_plugin
from handlers.tts_handler import text_to_speech
from handlers.llm_handler import generate_ai_response

# Global connections storage
websocket_connections: Dict[str, WebSocket] = {}

class VADProcessor:
    """WebRTC VAD processor for accurate voice detection"""
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 20):
        # VAD aggressiveness: 0-3 (1 is less strict, better for natural speech)
        self.vad = webrtcvad.Vad(mode=1)  # Giảm từ 2 xuống 1 để ít strict hơn
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # milliseconds
        self.frame_size = int(sample_rate * frame_duration / 1000)  # samples per frame
        
        # State tracking
        self.speech_frames = []
        self.silence_count = 0
        self.speech_count = 0
        self.is_speaking = False
        
        # Thresholds - tăng thời gian chờ để tránh ngắt giữa chừng
        self.silence_threshold = 75   # frames (~1.5 giây at 20ms/frame) - tăng từ 20
        self.speech_threshold = 2     # frames (~40ms at 20ms/frame) - giảm để nhạy hơn
        
        # Max speech duration (tránh quá dài)
        self.max_speech_frames = 15000  # ~5 phút at 20ms/frame
        
        print(f"[VAD] Initialized - Silence timeout: {self.silence_threshold * 20}ms")

    def process_audio(self, audio_data: bytes) -> tuple[bool, bool, bool]:
        """
        Process audio chunk and return (voice_detected, speech_started, speech_ended)
        """
        try:
            # Ensure audio is correct length for VAD
            expected_bytes = self.frame_size * 2
            if len(audio_data) != expected_bytes:
                if len(audio_data) < expected_bytes:
                    audio_data = audio_data + b'\x00' * (expected_bytes - len(audio_data))
                else:
                    audio_data = audio_data[:expected_bytes]
            
            # Detect voice in this frame
            is_voice = self.vad.is_speech(audio_data, self.sample_rate)
            
            speech_started = False
            speech_ended = False
            
            if is_voice:
                self.speech_count += 1
                self.silence_count = 0
                
                # Start speaking if enough consecutive speech frames
                if not self.is_speaking and self.speech_count >= self.speech_threshold:
                    self.is_speaking = True
                    speech_started = True
                    self.speech_frames = []  # Start fresh buffer
                    print(f"[VAD] Speech started (after {self.speech_count} frames)")
                
                # Add to speech buffer if speaking
                if self.is_speaking:
                    self.speech_frames.append(audio_data)
                    
                    # Kiểm tra max duration
                    if len(self.speech_frames) >= self.max_speech_frames:
                        print(f"[VAD] Max speech duration reached, forcing end")
                        speech_ended = True
                        self.is_speaking = False
                    
            else:
                self.speech_count = 0
                self.silence_count += 1
                
                # End speaking if enough consecutive silence frames
                if self.is_speaking and self.silence_count >= self.silence_threshold:
                    # Kiểm tra xem có đủ speech frames không (tránh chỉ có noise)
                    min_speech_frames = 10  # Tối thiểu 200ms speech
                    if len(self.speech_frames) >= min_speech_frames:
                        self.is_speaking = False
                        speech_ended = True
                        silence_duration = self.silence_count * self.frame_duration
                        print(f"[VAD] Speech ended - {len(self.speech_frames)} frames, {silence_duration}ms silence")
                    else:
                        # Quá ít speech frames, coi như chỉ là noise
                        print(f"[VAD] Discarding short speech ({len(self.speech_frames)} frames) - likely noise")
                        self.is_speaking = False
                        self.speech_frames = []  # Clear noise
                        # Không set speech_ended = True để không gửi vào STT
                        
                elif self.is_speaking and self.silence_count % 25 == 0:  # Debug mỗi 500ms
                    remaining = (self.silence_threshold - self.silence_count) * self.frame_duration
                    print(f"[VAD] Silence: {self.silence_count * self.frame_duration}ms, remaining: {remaining}ms")
            
            return is_voice, speech_started, speech_ended
            
        except Exception as e:
            print(f"[VAD] Processing error: {e}")
            return False, False, False
    
    def get_speech_audio(self) -> bytes:
        """Get accumulated speech audio and clear buffer"""
        audio = b''.join(self.speech_frames)
        self.speech_frames = []
        return audio

class ConversationState:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.conversation_id = f"conv_{agent_name}_{time.time()}"
        self.vad_processor = VADProcessor()
        self.is_agent_speaking = False
        self.should_interrupt = False
        self.tts_task: Optional[asyncio.Task] = None

async def handle_init_conversation(websocket: WebSocket, data: Dict[str, Any]) -> ConversationState:
    """Handle conversation initialization"""
    agent_name = data.get("agent_name", "default_agent")
    config = data.get("conversation_config_override", {})
    first_message = config.get("agent", {}).get("first_message", "Hello! How can I help you?")
    
    state = ConversationState(agent_name)
    
    # Send ready signal
    await websocket.send_text(json.dumps({
        "type": "conversation_initiation_metadata",
        "conversation_id": state.conversation_id,
    }))
    
    # Send first message
    if first_message:
        await send_agent_response(websocket, first_message, state)
    
    return state

async def send_agent_response(websocket: WebSocket, text: str, state: ConversationState):
    """Send agent response with TTS"""
    try:
        state.is_agent_speaking = True
        state.should_interrupt = False
        
        # Send text response
        await websocket.send_text(json.dumps({
            "type": "agent_response",
            "agent_response_event": {"agent_response": text}
        }))
        
        # Create and await TTS task
        state.tts_task = asyncio.create_task(stream_tts_chunks(websocket, text, state))
        await state.tts_task
        
    except asyncio.CancelledError:
        print("[TTS] Cancelled by user interrupt")
    finally:
        state.is_agent_speaking = False
        state.tts_task = None

async def stream_tts_chunks(websocket: WebSocket, text: str, state: ConversationState):
    """Stream TTS in chunks with interruption support"""
    chunk_size = 100
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        if state.should_interrupt:
            print(f"[TTS] Interrupted at chunk {i+1}/{len(chunks)}")
            return
            
        try:
            # Generate TTS
            audio_b64 = await text_to_speech(chunk, state.agent_name)
            
            if state.should_interrupt:
                return
                
            if audio_b64:
                await websocket.send_text(json.dumps({
                    "type": "audio",
                    "audio_event": {"audio_base_64": audio_b64}
                }))
                
                # Brief pause between chunks
                for _ in range(5):  # 50ms total
                    if state.should_interrupt:
                        return
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            print(f"[TTS] Error: {e}")
            return

async def interrupt_agent(state: ConversationState):
    """Interrupt agent speaking immediately"""
    if state.is_agent_speaking and not state.should_interrupt:
        print("[Interrupt] User speaking - stopping agent")
        state.should_interrupt = True
        
        if state.tts_task and not state.tts_task.done():
            state.tts_task.cancel()

async def handle_speech_complete(websocket: WebSocket, state: ConversationState, model_instance):
    """Process completed speech"""
    speech_audio = state.vad_processor.get_speech_audio()
    if not speech_audio:
        return
        
    try:
        # Speech to text
        transcript = await speech_to_text_with_plugin(speech_audio, state.agent_name)
        
        if transcript and transcript.strip():
            print(f"[STT] {transcript}")
            
            # Send transcript
            await websocket.send_text(json.dumps({
                "type": "user_transcript",
                "user_transcription_event": {"user_transcript": transcript}
            }))
            
            # Generate AI response
            # ai_response = await generate_ai_response(
            #     transcript, state.agent_name, state.conversation_id, model_instance
            # )
            ai_response = "It's interesting you mentioned the queen and the sister pair—sounds like a mystery! If you're Jessica and you've been watching through a cold one, maybe you're part of a larger story or a group with a secret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are you looking for help with the queen's case, or do you need assistance with the sister duo's investigationret? Are"
            # Send response
            await send_agent_response(websocket, ai_response, state)
            
    except Exception as e:
        print(f"[Speech] Error: {e}")

async def handle_audio_chunk(websocket: WebSocket, data: Dict[str, Any], 
                           state: ConversationState, model_instance):
    """Handle audio chunk with VAD processing"""
    audio_b64 = data.get("user_audio_chunk")
    if not audio_b64:
        return
        
    try:
        audio_data = base64.b64decode(audio_b64)
        
        # Process with WebRTC VAD
        voice_detected, speech_started, speech_ended = state.vad_processor.process_audio(audio_data)
        
        # INTERRUPT NGAY KHI PHÁT HIỆN VOICE - không cần chờ speaking confirmation
        if voice_detected:
            await interrupt_agent(state)
        
        # Handle speech completion - chỉ xử lý khi thực sự kết thúc
        if speech_ended:
            await handle_speech_complete(websocket, state, model_instance)
            
    except Exception as e:
        print(f"[Audio] Error: {e}")

# WebSocket endpoint
async def websocket_conversation_endpoint_enhanced(websocket: WebSocket, model_instance):
    """Enhanced websocket endpoint with WebRTC VAD"""
    await websocket.accept()
    
    conversation_state = None
    connection_id = f"conn_{time.time()}"
    
    try:
        websocket_connections[connection_id] = websocket
        
        async for message in websocket.iter_text():
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "conversation_initiation_client_data":
                conversation_state = await handle_init_conversation(websocket, data)
                
            elif message_type == "user_audio_chunk" and conversation_state:
                await handle_audio_chunk(websocket, data, conversation_state, model_instance)
                
            elif message_type == "stop_agent_speaking" and conversation_state:
                await interrupt_agent(conversation_state)

    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        # Cleanup
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]
        if conversation_state and conversation_state.tts_task:
            conversation_state.tts_task.cancel()
        await websocket.close()

def cleanup_connection(connection_id: str):
    """Clean up websocket connections"""
    if connection_id in websocket_connections:
        del websocket_connections[connection_id]

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

            for connection_id, connection_info in websocket_connections.items():
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