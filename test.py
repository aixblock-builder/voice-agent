import asyncio
import websockets
import pyaudio
import base64
import json
import threading
import queue
import signal
import sys
import wave
from io import BytesIO
import numpy as np

WS_URL = "ws://162.193.169.187:31604/conversation"

# Audio settings
CHUNK = 320  # 20ms at 16kHz (320 samples)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class RealTimeAudioClient:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = queue.Queue()
        self.playback_queue = queue.Queue()
        self.is_recording = False
        self.is_playing = False
        self.ws = None
        self.playback_thread = None
        
    def list_audio_devices(self):
        """List available audio devices to help choose the right ones"""
        print("📱 Available Audio Devices:")
        print("-" * 60)
        
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            device_type = []
            if info['maxInputChannels'] > 0:
                device_type.append("INPUT")
            if info['maxOutputChannels'] > 0:
                device_type.append("OUTPUT")
            
            print(f"Device {i}: {info['name']}")
            print(f"  Type: {' & '.join(device_type)}")
            print(f"  Max Channels: In={info['maxInputChannels']}, Out={info['maxOutputChannels']}")
            print(f"  Default Sample Rate: {info['defaultSampleRate']}")
            print()
        
    def get_best_devices(self):
        """Automatically select best input and output devices"""
        input_device = None
        output_device = None
        
        # Try to find dedicated microphone (not "Stereo Mix" or similar)
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            name_lower = info['name'].lower()
            
            # For input - prioritize actual microphones, avoid system audio capture
            if (info['maxInputChannels'] > 0 and input_device is None and
                'microphone' in name_lower or 'mic' in name_lower or 
                ('default' in name_lower and 'stereo mix' not in name_lower and 
                 'what u hear' not in name_lower and 'speakers' not in name_lower)):
                input_device = i
                
            # For output - prioritize speakers/headphones
            if (info['maxOutputChannels'] > 0 and output_device is None and
                ('speakers' in name_lower or 'headphones' in name_lower or 
                 'default' in name_lower)):
                output_device = i
        
        # Fallback to system defaults
        if input_device is None:
            try:
                input_device = self.audio.get_default_input_device_info()['index']
            except:
                input_device = 0
                
        if output_device is None:
            try:
                output_device = self.audio.get_default_output_device_info()['index']
            except:
                output_device = 0
        
        return input_device, output_device
        
    async def connect(self):
        """Connect to WebSocket server"""
        self.ws = await websockets.connect(WS_URL)
        print("✅ Connected to WebSocket server")
        
        # Send initialization
        init_msg = {
            "type": "conversation_initiation_client_data",
            "agent_name": "my_agent",
            "conversation_config_override": {
                "agent": {
                    "prompt": {"prompt": "You are a helpful assistant"},
                    "first_message": "Hello! I'm Jessica, please tell me anythin, listening..."
                }
            }
        }
        await self.ws.send(json.dumps(init_msg))
        print("📤 Sent conversation initiation")

    def setup_audio_streams(self):
        """Setup both input and output audio streams with device selection"""
        # List devices for debugging
        # self.list_audio_devices()  # Uncomment to see all devices
        
        # Get best devices automatically
        input_device, output_device = self.get_best_devices()
        
        input_info = self.audio.get_device_info_by_index(input_device)
        output_info = self.audio.get_device_info_by_index(output_device)
        
        print(f"🎤 Selected Input Device: {input_info['name']}")
        print(f"🔊 Selected Output Device: {output_info['name']}")
        
        # Input stream - chỉ từ microphone được chọn
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device,  # Chỉ định device cụ thể
            frames_per_buffer=CHUNK,
            stream_callback=self.audio_input_callback
        )
        
        # Output stream - chỉ ra speakers/headphones được chọn
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            output_device_index=output_device,  # Chỉ định device cụ thể
            frames_per_buffer=CHUNK
        )
        
        print("🎧 Audio streams ready with separate devices")

    def audio_input_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback - chỉ ghi khi không phát audio"""
        # Chỉ ghi âm khi:
        # 1. Đang trong chế độ recording
        # 2. KHÔNG đang phát audio từ server (tránh feedback)
        if self.is_recording and not self.is_playing:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def start_playback_thread(self):
        """Start the audio playback thread"""
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        print("🔊 Audio playback thread started")

    def _playback_worker(self):
        """Worker thread - phát audio và tạm dừng recording khi phát"""
        while True:
            try:
                # Chờ audio từ server
                audio_data = self.playback_queue.get()
                if audio_data is None:  # Signal to stop
                    break
                
                # TẠM DỪNG recording khi phát audio để tránh feedback
                self.is_playing = True
                print("🔇 Muting microphone during playback...")
                
                # Phát audio
                self._play_audio_immediately(audio_data)
                
                # Chờ một chút sau khi phát xong để tránh echo
                asyncio.run(asyncio.sleep(0.1))
                
                # BẬT LẠI recording
                self.is_playing = False
                print("🎤 Microphone active again")
                
            except Exception as e:
                print(f"❌ Playback error: {e}")
                self.is_playing = False

    def _play_audio_immediately(self, audio_data):
        """Phát audio ngay lập tức"""
        try:
            # Xử lý WAV format nếu có
            if audio_data.startswith(b'RIFF'):
                with BytesIO(audio_data) as wav_io:
                    with wave.open(wav_io, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        # Phát từng chunk nhỏ để giảm latency
                        chunk_size = CHUNK * 2  # 2 bytes per sample (16-bit)
                        for i in range(0, len(frames), chunk_size):
                            chunk = frames[i:i + chunk_size]
                            if chunk:
                                self.output_stream.write(chunk)
            else:
                # Raw audio data
                chunk_size = CHUNK * 2
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    if chunk:
                        self.output_stream.write(chunk)
                        
        except Exception as e:
            print(f"❌ Play error: {e}")

    async def listen_server(self):
        """Lắng nghe server - xử lý tất cả message types"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type", "")
                
                if msg_type == "ping":
                    # Ping/Pong heartbeat
                    event_id = data.get("ping_event", {}).get("event_id", "")
                    pong_msg = {"type": "pong", "event_id": event_id}
                    await self.ws.send(json.dumps(pong_msg))
                    
                elif msg_type == "audio":
                    # Audio từ server - phát ngay
                    audio_b64 = data.get("audio_event", {}).get("audio_base_64", "")
                    if audio_b64:
                        try:
                            audio_bytes = base64.b64decode(audio_b64)
                            # Queue để phát ngay
                            self.playback_queue.put(audio_bytes)
                            print("🎵 Playing server audio...")
                        except Exception as e:
                            print(f"❌ Audio decode error: {e}")
                        
                elif msg_type == "agent_response":
                    # Text response từ agent
                    response = data.get("agent_response_event", {}).get("agent_response", "")
                    if response:
                        print(f"🤖 Agent: {response}")
                    
                elif msg_type == "user_transcript":
                    # Transcript của user
                    transcript = data.get("user_transcription_event", {}).get("user_transcript", "")
                    if transcript:
                        print(f"👤 You said: {transcript}")
                    
                elif msg_type == "conversation_initiation_server_data":
                    print("✅ Conversation ready!")
                    
        except websockets.ConnectionClosed:
            print("❌ Connection closed")
        except Exception as e:
            print(f"❌ Server listen error: {e}")

    def start_recording(self):
        """Bắt đầu ghi âm liên tục"""
        self.is_recording = True
        self.input_stream.start_stream()
        print("🎤 Microphone recording started (will pause during playback)")

    def stop_recording(self):
        """Dừng ghi âm và cleanup"""
        self.is_recording = False
        self.is_playing = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        # Stop playback thread
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_queue.put(None)
            
        print("🛑 All audio stopped")

    async def send_audio_continuously(self):
        """Gửi audio chunks liên tục 20ms"""
        try:
            while self.is_recording:
                try:
                    # Lấy audio chunk từ queue
                    audio_data = self.audio_queue.get_nowait()
                    
                    # Encode base64
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    # Gửi lên server
                    audio_msg = {
                        "type": "user_audio_chunk",
                        "user_audio_chunk": audio_b64
                    }
                    
                    if not self.ws.closed:
                        await self.ws.send(json.dumps(audio_msg))
                    else:
                        print("❌ WebSocket closed")
                        break
                        
                except queue.Empty:
                    # Không có audio data, tiếp tục
                    pass
                except Exception as e:
                    print(f"❌ Send error: {e}")
                    
                # Chờ 20ms cho chunk tiếp theo
                await asyncio.sleep(0.02)
                
        except Exception as e:
            print(f"❌ Continuous send error: {e}")

    async def run(self):
        """Main loop - chạy tất cả tasks đồng thời"""
        try:
            # Setup audio
            self.setup_audio_streams()
            self.start_playback_thread()
            
            # Connect server
            await self.connect()
            
            # Start recording ngay
            self.start_recording()
            
            # Chạy đồng thời:
            # 1. Lắng nghe server
            # 2. Gửi audio liên tục
            await asyncio.gather(
                self.listen_server(),
                self.send_audio_continuously()
            )
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
        finally:
            # Cleanup
            self.stop_recording()
            if self.ws and not self.ws.closed:
                await self.ws.close()
            self.audio.terminate()

def signal_handler(sig, frame):
    """Graceful shutdown"""
    print('\n🛑 Shutting down...')
    sys.exit(0)

async def main():
    signal.signal(signal.SIGINT, signal_handler)
    client = RealTimeAudioClient()
    await client.run()

if __name__ == "__main__":
    print("🎙️ Real-time Conversation Client (Echo Prevention)")
    print("=" * 50)
    print("🎤 Microphone input only (no system audio)")  
    print("🔇 Auto-mutes mic during server audio playback")
    print("🔊 Separate input/output devices")
    print("📤 Sending 20ms chunks to server")
    print("💬 Like a real conversation without feedback!")
    print("=" * 50)
    print("Press Ctrl+C to stop")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Conversation ended!")
    except Exception as e:
        print(f"❌ Error: {e}")