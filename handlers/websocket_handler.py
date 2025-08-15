import asyncio
import json
import base64
import time
import re
import webrtcvad
from typing import Dict, Any, Optional, List
from starlette.websockets import WebSocket
from handlers.stt_handler import speech_to_text_with_plugin
from handlers.tts_handler import text_to_speech
from handlers.llm_handler import generate_ai_response

# Global connections storage
websocket_connections: Dict[str, WebSocket] = {}

class TTSQueueItem:
    def __init__(self, text: str, audio_b64: str, duration: float):
        self.text = text
        self.audio_b64 = audio_b64
        self.duration = duration

class VADProcessor:
    """WebRTC VAD processor with improved stability"""
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 20):
        self.vad = webrtcvad.Vad(mode=2)  # Mode 2 for better balance
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        self.speech_frames = []
        self.silence_count = 0
        self.speech_count = 0
        self.is_speaking = False
        
        # Tối ưu threshold cho real-time response
        self.silence_threshold = 40  # Giảm từ 75 xuống 40 (800ms -> 400ms)
        self.speech_threshold = 3    # Tăng từ 2 lên 3 để ít false positive
        self.max_speech_frames = 15000
        
        # Buffer để smooth detection
        self.voice_history = []
        self.history_size = 5
        
        print(f"[VAD] Initialized - Silence timeout: {self.silence_threshold * 20}ms, Speech threshold: {self.speech_threshold * 20}ms")

    def process_audio(self, audio_data: bytes) -> tuple[bool, bool, bool]:
        try:
            expected_bytes = self.frame_size * 2
            if len(audio_data) != expected_bytes:
                if len(audio_data) < expected_bytes:
                    audio_data = audio_data + b'\x00' * (expected_bytes - len(audio_data))
                else:
                    audio_data = audio_data[:expected_bytes]
            
            # WebRTC VAD detection
            raw_voice = self.vad.is_speech(audio_data, self.sample_rate)
            
            # Smooth detection với history buffer
            self.voice_history.append(raw_voice)
            if len(self.voice_history) > self.history_size:
                self.voice_history.pop(0)
            
            # Voice detected nếu ít nhất 60% frames gần đây có voice
            voice_ratio = sum(self.voice_history) / len(self.voice_history)
            is_voice = voice_ratio >= 0.6
            
            speech_started = speech_ended = False
            
            if is_voice:
                self.speech_count += 1
                self.silence_count = 0
                
                if not self.is_speaking and self.speech_count >= self.speech_threshold:
                    self.is_speaking = True
                    speech_started = True
                    self.speech_frames = []
                    print(f"[VAD] 🎤 Speech started (confidence: {voice_ratio:.2f})")
                
                if self.is_speaking:
                    self.speech_frames.append(audio_data)
                    if len(self.speech_frames) >= self.max_speech_frames:
                        speech_ended = True
                        self.is_speaking = False
                        print(f"[VAD] ⏹️  Max frames reached")
            else:
                self.speech_count = 0
                self.silence_count += 1
                
                if self.is_speaking and self.silence_count >= self.silence_threshold:
                    min_speech_frames = 8  # Giảm từ 10 xuống 8
                    if len(self.speech_frames) >= min_speech_frames:
                        self.is_speaking = False
                        speech_ended = True
                        duration_ms = len(self.speech_frames) * 20
                        print(f"[VAD] ✅ Speech ended - {duration_ms}ms ({len(self.speech_frames)} frames)")
                    else:
                        print(f"[VAD] 🗑️  Discarding short noise - {len(self.speech_frames)} frames")
                        self.is_speaking = False
                        self.speech_frames = []
            
            return is_voice, speech_started, speech_ended
            
        except Exception as e:
            print(f"[VAD] ❌ Error: {e}")
            return False, False, False
    
    def get_speech_audio(self) -> bytes:
        audio = b''.join(self.speech_frames)
        self.speech_frames = []
        return audio

class ConversationState:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.conversation_id = f"conv_{agent_name}_{time.time()}"
        self.vad_processor = VADProcessor()
        
        # TTS Queue system
        self.tts_queue: List[TTSQueueItem] = []
        self.is_playing = False
        self.should_interrupt = False
        self.queue_task: Optional[asyncio.Task] = None
        self.current_play_task: Optional[asyncio.Task] = None
        
        # Interrupt management
        self.interrupt_lock = asyncio.Lock()
        self.last_interrupt_time = 0

async def send_agent_response_direct(websocket: WebSocket, text: str, state: ConversationState):
    """Send agent response trực tiếp như logic cũ (cho first_message)"""
    try:
        state.is_playing = True
        state.should_interrupt = False
        
        # Send text response
        await websocket.send_text(json.dumps({
            "type": "agent_response",
            "agent_response_event": {"agent_response": text}
        }))
        
        # Stream TTS chunks như logic cũ
        await stream_tts_chunks_direct(websocket, text, state)
        
    except asyncio.CancelledError:
        print("[TTS Direct] 🛑 Cancelled by user interrupt")
    finally:
        state.is_playing = False

async def stream_tts_chunks_direct(websocket: WebSocket, text: str, state: ConversationState):
    """Stream TTS chunks trực tiếp như logic cũ với frequent interrupt checking"""
    chunk_size = 100
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        # Check interrupt mỗi chunk
        if state.should_interrupt:
            print(f"[TTS Direct] 🛑 Interrupted at chunk {i+1}/{len(chunks)}")
            return
            
        try:
            # Generate TTS
            audio_b64 = await text_to_speech(chunk, state.agent_name)
            
            # Check interrupt sau khi generate xong
            if state.should_interrupt:
                print(f"[TTS Direct] 🛑 Interrupted after TTS generation")
                return
                
            if audio_b64:
                await websocket.send_text(json.dumps({
                    "type": "audio",
                    "audio_event": {"audio_base_64": audio_b64}
                }))
                
                # Brief pause với interrupt checking
                for j in range(10):  # 100ms total, check mỗi 10ms
                    if state.should_interrupt:
                        print(f"[TTS Direct] 🛑 Interrupted during pause")
                        return
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            print(f"[TTS Direct] ❌ Error: {e}")
            return

def split_into_sentences(text: str) -> List[str]:
    """Tách text thành các câu"""
    sentences = re.split(r'[.!?]+(?:\s+|$)', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def calculate_audio_duration(audio_b64: str, sample_rate: int = 24000) -> float:
    """Tính thời gian audio từ base64 với accuracy cao hơn"""
    try:
        audio_bytes = base64.b64decode(audio_b64)
        # Giả định 16-bit PCM
        num_samples = len(audio_bytes) // 2
        duration = num_samples / sample_rate
        return max(duration, 0.1)  # Minimum 0.1s
    except:
        # Fallback dựa trên length
        return max(len(audio_b64) * 0.0008, 0.1)

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
    
    # Send first message với logic cũ (gửi ngay không qua queue)
    if first_message:
        await send_agent_response_direct(websocket, first_message, state)
    
    return state

async def queue_agent_response(websocket: WebSocket, text: str, state: ConversationState):
    """Queue agent response với TTS và immediate interrupt checking"""
    try:
        # Check interrupt trước khi bắt đầu
        if state.should_interrupt:
            print("[TTS Queue] 🛑 Interrupted before queuing")
            return
            
        # Send text response ngay
        await websocket.send_text(json.dumps({
            "type": "agent_response",
            "agent_response_event": {"agent_response": text}
        }))
        
        # Tách thành sentences
        sentences = split_into_sentences(text)
        print(f"[TTS Queue] 📝 Queuing {len(sentences)} sentences")
        
        for i, sentence in enumerate(sentences):
            # Check interrupt giữa mỗi sentence
            if state.should_interrupt:
                print(f"[TTS Queue] 🛑 Interrupted while generating sentence {i+1}/{len(sentences)}")
                return
                
            if not sentence:
                continue
                
            # Generate TTS
            audio_b64 = await text_to_speech(sentence, state.agent_name)
            
            # Check interrupt sau TTS generation
            if state.should_interrupt:
                print(f"[TTS Queue] 🛑 Interrupted after TTS gen for sentence {i+1}")
                return
                
            if audio_b64:
                duration = calculate_audio_duration(audio_b64)
                queue_item = TTSQueueItem(sentence, audio_b64, duration)
                state.tts_queue.append(queue_item)
        
        # Start queue processor nếu chưa chạy và không bị interrupt
        if not state.should_interrupt and (not state.queue_task or state.queue_task.done()):
            state.queue_task = asyncio.create_task(process_tts_queue(websocket, state))
            
    except Exception as e:
        print(f"[TTS Queue] ❌ Error: {e}")

async def process_tts_queue(websocket: WebSocket, state: ConversationState):
    """Xử lý hàng chờ TTS với ultra-responsive interrupt checking"""
    try:
        print(f"[TTS Queue] 🚀 Starting queue processor with {len(state.tts_queue)} items")
        
        while state.tts_queue and not state.should_interrupt:
            queue_item = state.tts_queue.pop(0)
            
            # Final check trước khi play
            if state.should_interrupt:
                print("[TTS Queue] 🛑 Interrupted before playing item")
                break
            
            state.is_playing = True
            
            # Send audio
            await websocket.send_text(json.dumps({
                "type": "audio",
                "audio_event": {"audio_base_64": queue_item.audio_b64}
            }))
            
            print(f"[TTS Queue] 🔊 Playing: '{queue_item.text[:40]}...' ({queue_item.duration:.2f}s)")
            
            # Ultra-frequent interrupt checking during playback
            total_wait_time = 0
            check_interval = 0.02  # 20ms - matching client chunk rate!
            
            while total_wait_time < queue_item.duration:
                if state.should_interrupt:
                    print(f"[TTS Queue] 🛑 INTERRUPTED at {total_wait_time:.2f}s/{queue_item.duration:.2f}s")
                    state.is_playing = False
                    state.tts_queue.clear()  # Clear remaining queue
                    return
                    
                await asyncio.sleep(check_interval)
                total_wait_time += check_interval
            
            state.is_playing = False
            print(f"[TTS Queue] ✅ Completed: '{queue_item.text[:30]}...'")
            
            # Ultra-short gap với frequent checking
            gap_time = 0
            gap_duration = 0.1  # Giảm xuống 100ms
            while gap_time < gap_duration:
                if state.should_interrupt:
                    print("[TTS Queue] 🛑 Interrupted during gap")
                    state.tts_queue.clear()
                    return
                await asyncio.sleep(0.02)  # 20ms checks
                gap_time += 0.02
            
        print(f"[TTS Queue] 🏁 Queue finished. Remaining: {len(state.tts_queue)}")
            
    except asyncio.CancelledError:
        print("[TTS Queue] 🛑 Task cancelled")
    except Exception as e:
        print(f"[TTS Queue] ❌ Error: {e}")
    finally:
        state.is_playing = False
        state.tts_queue.clear()

async def interrupt_agent(state: ConversationState):
    """Ngắt agent với debouncing để tránh spam"""
    async with state.interrupt_lock:
        current_time = time.time()
        
        # Debounce: chỉ process interrupt nếu đã > 100ms từ lần trước
        if current_time - state.last_interrupt_time < 0.1:
            return
            
        state.last_interrupt_time = current_time
        
        if (state.is_playing or state.tts_queue) and not state.should_interrupt:
            print("[Interrupt] 🛑 USER SPEAKING - IMMEDIATE STOP")
            state.should_interrupt = True
            
            # Clear queue ngay lập tức
            state.tts_queue.clear()
            
            # Cancel current tasks
            if state.queue_task and not state.queue_task.done():
                state.queue_task.cancel()
            
            # Reset state sau delay ngắn
            asyncio.create_task(reset_interrupt_flag(state))

async def reset_interrupt_flag(state: ConversationState):
    """Reset interrupt flag sau khi cleanup"""
    await asyncio.sleep(0.3)  # Giảm từ 0.5 xuống 0.3
    state.should_interrupt = False
    print("[Interrupt] 🔄 Ready for new responses")

async def handle_speech_complete(websocket: WebSocket, state: ConversationState, model_instance):
    """Process completed speech với improved error handling"""
    speech_audio = state.vad_processor.get_speech_audio()
    if not speech_audio:
        return
        
    audio_duration = len(speech_audio) * 20 / 1000  # ms to seconds
    print(f"[Speech] 🎯 Processing {len(speech_audio)} bytes ({audio_duration:.1f}s)")
        
    try:
        # Speech to text
        transcript = await speech_to_text_with_plugin(speech_audio, state.agent_name)
        
        if transcript and transcript.strip():
            print(f"[STT] 📝 '{transcript}'")
            
            # Send transcript
            await websocket.send_text(json.dumps({
                "type": "user_transcript", 
                "user_transcription_event": {"user_transcript": transcript}
            }))
            
            # Generate AI response
            # ai_response = await generate_ai_response(
            #     transcript, state.agent_name, state.conversation_id, model_instance
            # )
            ai_response = f"I heard you say: '{transcript}'. That's interesting! Let me help you with that. Thank you for your question."
            
            # Queue response với interrupt checking
            await queue_agent_response(websocket, ai_response, state)
        else:
            print("[STT] 🔇 No valid transcript received")
            
    except Exception as e:
        print(f"[Speech] ❌ Processing error: {e}")

async def handle_audio_chunk(websocket: WebSocket, data: Dict[str, Any], 
                           state: ConversationState, model_instance):
    """Handle audio chunk với optimized interrupt logic"""
    audio_b64 = data.get("user_audio_chunk")
    if not audio_b64:
        return
        
    try:
        audio_data = base64.b64decode(audio_b64)
        
        # Process với WebRTC VAD
        voice_detected, speech_started, speech_ended = state.vad_processor.process_audio(audio_data)
        
        # CRITICAL: Interrupt IMMEDIATELY khi detect voice
        if voice_detected and (state.is_playing or state.tts_queue):
            # Don't await - fire and forget để không block audio processing
            asyncio.create_task(interrupt_agent(state))
        
        if speech_started:
            print("[VAD] 🎤 Speech detection started - interrupting agent")
            
        # Handle speech completion
        if speech_ended:
            print("[VAD] ⏹️  Speech ended, processing...")
            # Process trong background để không block audio stream
            asyncio.create_task(handle_speech_complete(websocket, state, model_instance))
            
    except Exception as e:
        print(f"[Audio] ❌ Processing error: {e}")

async def websocket_conversation_endpoint_enhanced(websocket: WebSocket, model_instance):
    """Enhanced websocket endpoint với ultra-responsive interrupt"""
    await websocket.accept()
    print("[WebSocket] 🔌 Connection accepted")
    
    conversation_state = None
    connection_id = f"conn_{time.time()}"
    
    try:
        websocket_connections[connection_id] = websocket
        
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                message_type = data.get("type")

                if message_type == "conversation_initiation_client_data":
                    conversation_state = await handle_init_conversation(websocket, data)
                    print(f"[WebSocket] 🚀 Conversation initialized: {conversation_state.conversation_id}")
                    
                elif message_type == "user_audio_chunk" and conversation_state:
                    # Handle audio chunks immediately - no queuing
                    await handle_audio_chunk(websocket, data, conversation_state, model_instance)
                    
                elif message_type == "stop_agent_speaking" and conversation_state:
                    print("[WebSocket] 🛑 Manual stop requested")
                    await interrupt_agent(conversation_state)

            except json.JSONDecodeError:
                print("[WebSocket] ❌ Invalid JSON received")
            except Exception as e:
                print(f"[WebSocket] ❌ Message processing error: {e}")

    except Exception as e:
        print(f"[WebSocket] ❌ Connection error: {e}")
    finally:
        # Cleanup
        print(f"[WebSocket] 🔌 Connection {connection_id} closing")
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]
        if conversation_state:
            if conversation_state.queue_task and not conversation_state.queue_task.done():
                conversation_state.queue_task.cancel()
        try:
            await websocket.close()
        except:
            pass

def cleanup_connection(connection_id: str):
    """Clean up websocket connections"""
    if connection_id in websocket_connections:
        del websocket_connections[connection_id]

async def handle_pong(data: Dict[str, Any]):
    """Handle pong message"""
    event_id = data.get("event_id")
    print(f"[Agent] 🏓 Received pong for event: {event_id}")

async def send_ping_to_clients():
    """Send periodic ping to all connected clients"""
    while True:
        try:
            await asyncio.sleep(30)
            
            for connection_id, websocket in websocket_connections.items():
                if websocket:
                    event_id = f"ping_{time.time()}"
                    await websocket.send_text(
                        json.dumps(
                            {"type": "ping", "ping_event": {"event_id": event_id}}
                        )
                    )
        except Exception as e:
            print(f"[Ping] ❌ Error: {e}")