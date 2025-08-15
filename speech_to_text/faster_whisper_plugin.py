from .asr_base import AsrBase, register_plugin
from faster_whisper import WhisperModel
import base64
import numpy as np

@register_plugin("faster_whisper")
class FasterWhisperPlugin(AsrBase):
    def __init__(self, model_size: str = "base", **kwargs):
        super().__init__(**kwargs)
        self.model_size = model_size
        self.language = kwargs.get("language", None)

    def load(self):
        self.model = WhisperModel(self.model_size, device=self.device)
        return self
    
    def predict(self, audio):
        if self.model is None:
            self.load()

        # Handle file path, numpy array, or base64 bytes
        if isinstance(audio, str):
            if audio.startswith('data:') or len(audio) > 255:  # Base64 string
                # Decode base64 to bytes
                audio_bytes = base64.b64decode(audio)
                # Convert bytes to numpy array (assuming 16kHz, 16-bit PCM)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # File path - load and preprocess
                audio_data = self.preprocess(audio)
        elif isinstance(audio, bytes):
            # Raw bytes - convert to numpy array
            audio_data = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            # Assume it's numpy array or file path from converter
            audio_data = self.preprocess(audio)
        
        # Bảo đảm kiểu/shape đúng cho faster-whisper
        audio_data = np.asarray(audio_data, dtype=np.float32)
        if audio_data.ndim > 1:
            # Nếu là stereo (n, 2) → trộn về mono
            audio_data = audio_data.mean(axis=-1).astype(np.float32)

        if audio_data.size == 0:
            raise ValueError("Audio trống sau khi tiền xử lý.")

        # --- Gọi faster-whisper ---
        beam_size = getattr(self, "beam_size", 5)
        language = getattr(self, "language", None)  # None để tự detect
        vad_filter = getattr(self, "vad_filter", True)
        task = getattr(self, "task", "transcribe")  # hoặc "translate" nếu muốn

        segments, info = self.model.transcribe(
            audio_data,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
            task=task,
            # tuỳ chọn thêm:
            # temperature=0.0,
            # best_of=5,
            # condition_on_previous_text=True,
        )

        # Thu thập kết quả
        seg_list = []
        full_text_parts = []
        for i, seg in enumerate(segments):
            seg_dict = {
                "id": i,
                "start": float(seg.start) if seg.start is not None else None,
                "end": float(seg.end) if seg.end is not None else None,
                "text": seg.text.strip() if seg.text else "",
            }
            # Nếu bật word_timestamps trong model, có thể thêm:
            # seg_dict["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in (seg.words or [])]
            seg_list.append(seg_dict)
            if seg_dict["text"]:
                full_text_parts.append(seg_dict["text"])

        return " ".join(full_text_parts).strip()