import torch
import numpy as np
# from pydub import AudioSegment
import uuid
from silero_vad import load_silero_vad

# --- Các hằng số và model giữ nguyên ---
SAMPLE_RATE = 16000
FRAME_SIZE = 512
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

# def save_buffer_to_mp3(waveform: torch.Tensor, sample_rate: int) -> str:
#     """
#     Save a waveform tensor to an MP3 file using pydub.

#     Args:
#         waveform (torch.Tensor): Shape (1, num_samples)
#         sample_rate (int): Sampling rate

#     Returns:
#         str: Path to saved MP3 file
#     """
#     # Ensure mono channel
#     if waveform.shape[0] > 1:
#         waveform = waveform[0].unsqueeze(0)

#     # Convert to NumPy int16 (pydub expects this)
#     samples = (waveform.squeeze().numpy() * 32767).astype(np.int16)

#     # Convert to AudioSegment
#     audio_segment = AudioSegment(
#         samples.tobytes(),
#         frame_rate=sample_rate,
#         sample_width=2,  # 16-bit => 2 bytes
#         channels=1
#     )

#     filename = f"{uuid.uuid4()}.mp3"
#     audio_segment.export(filename, format="mp3")
#     return filename