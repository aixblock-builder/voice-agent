import asyncio
from itertools import tee
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
    """WebRTC VAD processor with improved stability and reset capability"""
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 20):
        self.vad = webrtcvad.Vad(mode=2)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        self.reset_state()
        
        # Optimized thresholds for real-time response
        self.silence_threshold = 25  # 500ms
        self.speech_threshold = 2    # 40ms
        self.max_speech_frames = 15000
        
        # Buffer for smooth detection
        self.history_size = 3
        
        print(f"[VAD] Initialized - Silence: {self.silence_threshold * 20}ms, Speech: {self.speech_threshold * 20}ms")

    def reset_state(self):
        """Reset VAD state completely"""
        self.speech_frames = []
        self.silence_count = 0
        self.speech_count = 0
        self.is_speaking = False
        self.voice_history = []
        print("[VAD] 🔄 State reset")

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
            
            # Smooth detection with history buffer
            self.voice_history.append(raw_voice)
            if len(self.voice_history) > self.history_size:
                self.voice_history.pop(0)
            
            # Voice detected if at least 50% of recent frames have voice
            voice_ratio = sum(self.voice_history) / len(self.voice_history)
            is_voice = voice_ratio >= 0.5
            
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
                    min_speech_frames = 6
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
        
        # TTS Queue system with improved interrupt handling
        self.tts_queue: List[TTSQueueItem] = []
        self.is_playing = False
        self.queue_task: Optional[asyncio.Task] = None
        
        # IMPROVED: Event-based interrupt system for instant response
        self._interrupt_event = asyncio.Event()
        self.interrupt_lock = asyncio.Lock()
        
        # Add processing lock to prevent race conditions
        self.processing_lock = asyncio.Lock()

    @property
    def should_interrupt(self) -> bool:
        """Immediate interrupt check"""
        return self._interrupt_event.is_set()
    
    def set_interrupt(self):
        """Set interrupt flag immediately"""
        self._interrupt_event.set()
        print(f"[Interrupt] ⚡ INTERRUPT FLAG SET")
    
    def clear_interrupt(self):
        """Clear interrupt flag"""
        self._interrupt_event.clear()
        print(f"[Interrupt] ✅ INTERRUPT FLAG CLEARED")

def split_into_smart_chunks(text: str, max_chunk_length: int = 40) -> List[str]:
    """Split text into smart chunks by words, not cutting mid-word"""
    # First split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If sentence is shorter than max_chunk_length, keep as is
        if len(sentence) <= max_chunk_length:
            chunks.append(sentence)
            continue
        
        # Split long sentences into chunks by words
        words = sentence.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Check if adding this word would exceed max_length
            word_length = len(word) + (1 if current_chunk else 0)  # +1 for space
            
            if current_length + word_length > max_chunk_length and current_chunk:
                # Finish current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                # Add word to current chunk
                current_chunk.append(word)
                current_length += word_length
        
        # Add final chunk if remaining
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    # Filter empty chunks
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def calculate_audio_duration(audio_b64: str, sample_rate: int = 24000) -> float:
    """Calculate audio duration from base64 with higher accuracy"""
    try:
        audio_bytes = base64.b64decode(audio_b64)
        # Assume 16-bit PCM
        num_samples = len(audio_bytes) // 2
        duration = num_samples / sample_rate
        return max(duration, 0.05)  # Minimum 50ms
    except:
        # Fallback based on length
        return max(len(audio_b64) * 0.0006, 0.05)

async def interrupt_agent_instant(state: ConversationState):
    """INSTANT interrupt with zero-delay response and proper cleanup"""
    print(f"[Interrupt] 🚨 IMMEDIATE INTERRUPT TRIGGERED")
    
    async with state.interrupt_lock:
        was_playing = state.is_playing
        queue_length = len(state.tts_queue)
        
        if (state.is_playing or state.tts_queue) and not state.should_interrupt:
            print(f"[Interrupt] ⚡ STOPPING: playing={was_playing}, queue={queue_length}")
            
            # Set interrupt flag immediately
            state.set_interrupt()
            
            # Clear queue immediately
            state.tts_queue.clear()
            state.is_playing = False  # Force stop playing state
            print(f"[Interrupt] 🧹 Cleared {queue_length} items + stopped playback")
            
            # Cancel current task if running
            if state.queue_task and not state.queue_task.done():
                print("[Interrupt] 🛑 Cancelling queue task")
                state.queue_task.cancel()
                try:
                    await state.queue_task
                except asyncio.CancelledError:
                    print("[Interrupt] 🛑 Task cancelled successfully")
            
            # Reset VAD state to ensure fresh detection
            state.vad_processor.reset_state()
            
            # Reset flag after short delay
            asyncio.create_task(reset_interrupt_flag(state))
        else:
            print(f"[Interrupt] ⏭️  No action needed - already interrupted or not playing")

async def reset_interrupt_flag(state: ConversationState):
    """Reset interrupt flag after short delay"""
    await asyncio.sleep(0.06)  # 60ms - even shorter
    state.clear_interrupt()

async def process_tts_queue_instant(websocket: WebSocket, state: ConversationState):
    """Improved TTS queue processor with frequent yield for audio processing"""
    try:
        print(f"[TTS Queue] 🚀 Starting with {len(state.tts_queue)} items")
        
        while state.tts_queue:
            # IMMEDIATE interrupt check before each item
            if state.should_interrupt:
                print("[TTS Queue] ⚡ INSTANT INTERRUPT - STOPPING")
                state.is_playing = False
                state.tts_queue.clear()
                return
            
            queue_item = state.tts_queue.pop(0)
            state.is_playing = True
            
            # Send audio chunk
            await websocket.send_text(json.dumps({
                "type": "audio",
                "audio_event": {"audio_base_64": queue_item.audio_b64}
            }))
            
            print(f"[TTS Queue] 🔊 Playing: '{queue_item.text[:40]}...' ({queue_item.duration:.2f}s)")
            
            # CRITICAL: Use shorter intervals with frequent yielding to event loop
            total_wait = 0
            interval = 0.01  # 10ms intervals for ultra-responsive interrupt
            
            while total_wait < queue_item.duration:
                # Check interrupt immediately
                if state.should_interrupt:
                    print(f"[TTS Queue] ⚡ INTERRUPTED during playback after {total_wait:.2f}s")
                    state.is_playing = False
                    state.tts_queue.clear()
                    return
                
                # Short sleep with yield to allow audio processing
                await asyncio.sleep(interval)
                total_wait += interval
                
                # Yield control to event loop more frequently
                if int(total_wait * 100) % 5 == 0:  # Every 50ms
                    await asyncio.sleep(0)  # Yield to event loop
            
            state.is_playing = False
            print(f"[TTS Queue] ✅ Completed: '{queue_item.text[:30]}...'")
            
            # Very short gap with immediate interrupt check
            if state.should_interrupt:
                print("[TTS Queue] ⚡ INTERRUPTED during gap")
                state.tts_queue.clear()
                return
                
            # Tiny gap with yield
            await asyncio.sleep(0.02)  # 20ms gap
            await asyncio.sleep(0)     # Yield to event loop
            
        print("[TTS Queue] 🏁 Queue completed normally")
        
    except asyncio.CancelledError:
        print("[TTS Queue] 🛑 Task cancelled")
        state.is_playing = False
        state.tts_queue.clear()
        raise
    except Exception as e:
        print(f"[TTS Queue] ❌ Error: {e}")
    finally:
        state.is_playing = False

async def send_agent_response_unified(websocket: WebSocket, text: str, state: ConversationState, is_first_message: bool = False):
    """Unified response handler with improved interrupt handling"""
    try:
        # Send text response immediately
        await websocket.send_text(json.dumps({
            "type": "agent_response",
            "agent_response_event": {"agent_response": text}
        }))
        
        # Split into smaller chunks for better interrupt granularity
        chunks = split_into_smart_chunks(text, max_chunk_length=40)
        print(f"[TTS] 📝 Processing {len(chunks)} chunks {'(FIRST MESSAGE)' if is_first_message else ''}")
        
        if is_first_message:
            # FIRST MESSAGE: Send immediately without queue
            for chunk in chunks:
                if not chunk:
                    continue
                    
                audio_b64 = await text_to_speech(chunk, state.agent_name)
                if audio_b64:
                    await websocket.send_text(json.dumps({
                        "type": "audio",
                        "audio_event": {"audio_base_64": audio_b64}
                    }))
                    print(f"[TTS First] ⚡ Sent immediately: '{chunk[:40]}...'")
        else:
            # NORMAL RESPONSES: Generate and queue with interrupt checking
            async with state.processing_lock:  # Prevent race conditions
                for i, chunk in enumerate(chunks):
                    # Check interrupt before each chunk generation
                    if state.should_interrupt:
                        print(f"[TTS] 🛑 Interrupted while generating chunk {i+1}/{len(chunks)}")
                        return
                        
                    if not chunk:
                        continue
                        
                    # Generate TTS for chunk
                    audio_b64 = await text_to_speech(chunk, state.agent_name)
                    
                    # Check interrupt after TTS generation
                    if state.should_interrupt:
                        print(f"[TTS] 🛑 Interrupted after TTS gen for chunk {i+1}")
                        return
                        
                    if audio_b64:
                        duration = calculate_audio_duration(audio_b64)
                        queue_item = TTSQueueItem(chunk, audio_b64, duration)
                        state.tts_queue.append(queue_item)
                        print(f"[TTS] ➕ Queued: '{chunk[:35]}...' ({duration:.2f}s)")
                
                # Start playing queue if not interrupted and no task running
                if not state.should_interrupt and (not state.queue_task or state.queue_task.done()):
                    state.queue_task = asyncio.create_task(process_tts_queue_instant(websocket, state))
            
    except Exception as e:
        print(f"[TTS] ❌ Error: {e}")

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
    
    # Send first message with immediate delivery (no queue)
    if first_message:
        await send_agent_response_unified(websocket, first_message, state, is_first_message=True)
    
    return state

async def handle_speech_complete(websocket: WebSocket, state: ConversationState, model_instance):
    """Process completed speech with improved error handling"""
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
            ai_response = "This is a shorter test response for better interrupt testing. It should be easier to interrupt with fewer chunks. Testing the new improved system now."
            
            # Use unified response system with improved interrupt handling
            await send_agent_response_unified(websocket, ai_response, state, is_first_message=False)
            
    except Exception as e:
        print(f"[Speech] ❌ Processing error: {e}")

async def handle_audio_chunk(websocket: WebSocket, data: Dict[str, Any], 
                           state: ConversationState, model_instance):
    """Handle audio chunk with instant interrupt and debug logging"""
    audio_b64 = data.get("user_audio_chunk")
    if not audio_b64:
        return
        
    try:
        audio_data = base64.b64decode(audio_b64)
        
        # Process with WebRTC VAD
        voice_detected, speech_started, speech_ended = state.vad_processor.process_audio(audio_data)
        
        # DEBUG: Log every voice detection for troubleshooting
        if voice_detected:
            playing_status = f"playing={state.is_playing}, queue={len(state.tts_queue)}, interrupted={state.should_interrupt}"
            print(f"[Audio] 🔊 VOICE! {playing_status}")
        
        # ⚡ CRITICAL: INSTANT interrupt on ANY voice detection during playback
        if voice_detected and (state.is_playing or state.tts_queue) and not state.should_interrupt:
            print(f"[Audio] 🚨 TRIGGERING INTERRUPT!")
            # Direct await instead of create_task for zero delay
            await interrupt_agent_instant(state)
        elif voice_detected and state.should_interrupt:
            print(f"[Audio] ⏭️  Voice detected but already interrupted")
        
        if speech_started:
            print("[VAD] 🎤 Speech started")
            
        # Handle speech completion
        if speech_ended:
            print("[VAD] ⏹️  Speech ended, processing transcript...")
            # Process in background to not block audio stream
            asyncio.create_task(handle_speech_complete(websocket, state, model_instance))
            
    except Exception as e:
        print(f"[Audio] ❌ Processing error: {e}")

async def websocket_conversation_endpoint_enhanced(websocket: WebSocket, model_instance):
    """Enhanced websocket endpoint with instant interrupt response"""
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
                    # ⚡ Process audio chunks immediately with instant interrupt
                    await handle_audio_chunk(websocket, data, conversation_state, model_instance)
                    
                elif message_type == "stop_agent_speaking" and conversation_state:
                    print("[WebSocket] 🛑 Manual stop requested")
                    await interrupt_agent_instant(conversation_state)

                elif message_type == "pong" and conversation_state:
                    await handle_pong(data)

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