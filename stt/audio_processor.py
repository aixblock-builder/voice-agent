# audio_processor.py
import webrtcvad
import numpy as np
from typing import List

class AudioProcessor:
    """
    - pipe  : Hugging Face Whisper pipeline (đã load sẵn)
    - mode  : 0 (hung hăng) → 3 (dễ dãi) cho webrtcvad
    - frame_ms        : chiều dài 1 khung phân tích (10/20/30 ms)
    - silence_frames  : số khung im lặng liên tiếp để kết thúc câu nói
    """
    def __init__(
        self,
        pipe,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        mode: int = 2,
        silence_frames: int = 10,        # 10 × 30 ms ≈ 0 ,3 s im lặng
        min_audio_ms: int = 500          # tối thiểu 0,5 s mới gửi Whisper
    ):
        self.pipe = pipe
        self.sr = sample_rate
        self.vad = webrtcvad.Vad(mode)
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 2 byte / sample
        self.silence_frames = silence_frames
        self.min_audio_bytes = int(sample_rate * min_audio_ms / 1000) * 2

        # Bộ đệm
        self._bytes_ring: bytearray = bytearray()   # gom dữ liệu từ WebSocket
        self._speech_buffer: List[bytes] = []       # các frame tiếng nói
        self._silence_count = 0
        self.processing = False

    # ---------- util ----------
    def _frames_from_bytes(self, new_bytes: bytes):
        """ Gom byte vào ring và yield các frame đủ size """
        self._bytes_ring.extend(new_bytes)
        while len(self._bytes_ring) >= self.frame_bytes:
            frame = bytes(self._bytes_ring[: self.frame_bytes])
            del self._bytes_ring[: self.frame_bytes]
            yield frame

    def _flush_to_whisper(self) -> str:
        """Gộp buffer, convert sang float32 và gọi Whisper"""
        raw = b"".join(self._speech_buffer)
        self._speech_buffer.clear()

        if len(raw) < self.min_audio_bytes:
            return ""   # đoạn quá ngắn → bỏ

        int16_audio = np.frombuffer(raw, dtype=np.int16)
        float32_audio = int16_audio.astype(np.float32) / 32768.0
        sample = {"array": float32_audio, "sampling_rate": self.sr}

        result = self.pipe(sample, return_timestamps=True, generate_kwargs={"language": "en"})
        return result["text"]

    # ---------- hàm chính ----------
    async def process_audio(self, chunk: bytes) -> str:
        """
        Trả về transcript khi xác định người dùng đã ngừng nói,
        nếu chưa đủ điều kiện sẽ trả "".
        """
        transcript = ""

        for frame in self._frames_from_bytes(chunk):
            is_speech = self.vad.is_speech(frame, self.sr)

            if is_speech:
                self._speech_buffer.append(frame)
                self._silence_count = 0
            else:
                # im lặng
                if self._speech_buffer:          # chỉ đếm im lặng sau khi từng nói
                    self._silence_count += 1

            # Nếu im lặng đủ lâu → flush
            if self._silence_count >= self.silence_frames and not self.processing:
                self.processing = True
                try:
                    transcript = self._flush_to_whisper()
                finally:
                    self._silence_count = 0
                    self.processing = False

        return transcript
