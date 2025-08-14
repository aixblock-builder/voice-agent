import torch
import numpy as np
# from pydub import AudioSegment
import uuid
from silero_vad import load_silero_vad

# --- Các hằng số và model giữ nguyên ---
SAMPLE_RATE = 16000
FRAME_SIZE = 512
BYTES_PER_SAMPLE = 2
FRAMES_PER_BUFFER = 10
VAD_SPEECH_THRESHOLD = 0.6
SILENCE_TIMEOUT_SECONDS = 1.5
MAX_RECORDING_SECONDS = 30
CHUNK_SIZE_BYTES = FRAME_SIZE * 2 * FRAMES_PER_BUFFER
SILENCE_FRAMES_THRESHOLD = int((SILENCE_TIMEOUT_SECONDS * SAMPLE_RATE) / FRAME_SIZE)
MAX_RECORDING_FRAMES = int((MAX_RECORDING_SECONDS * SAMPLE_RATE) / FRAME_SIZE)
BYTES_PER_SAMPLE = 2
vad_model = load_silero_vad()


# HÀM MỚI: Thay thế cho process_audio_buffer
def process_frame(frame_bytes: bytes) -> tuple[bool, torch.Tensor | None]:
    """
    Xử lý một frame âm thanh duy nhất (bytes) và thực hiện VAD.
    Đây là một hàm thuần túy, không có tác dụng phụ.
    """
    try:
        # Chuyển đổi bytes thành tensor float32 đã được chuẩn hóa
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)  # Shape: (1, FRAME_SIZE)
    except Exception as e:
        print(f"Lỗi giải mã khung âm thanh: {e}")
        return False, None

    # Chạy VAD
    try:
        # item() > 0.5 sẽ trả về True nếu có tiếng nói
        is_speech = vad_model(waveform, SAMPLE_RATE).item() > 0.5
    except Exception as e:
        print(f"Lỗi mô hình VAD: {e}")
        return False, waveform

    return is_speech, waveform
