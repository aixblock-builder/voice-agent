from .asr_base import AsrBase, register_plugin
import uuid
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
from pydub import AudioSegment
from silero_vad import load_silero_vad

@register_plugin("qwen_audio")
class QwenAudioPlugin(AsrBase):
    def __init__(self, model_size: str = "base", **kwargs):
        super().__init__(**kwargs)
        self.model_size = model_size
        self.language = kwargs.get("language", None)
        self.model = None
        self.processor = None
        self.SAMPLE_RATE = 16000
        self.FRAME_SIZE = 512
        self.BYTES_PER_SAMPLE = 2
        self.vad_model = load_silero_vad()

    def load(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
        return self
    
    def predict(self, audio):
        if self.model is None or self.processor is None:
            self.load()
    
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a speech recognition assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": path},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        audios = []
        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(librosa.load(ele["audio_url"], sr=self.processor.feature_extractor.sampling_rate)[0])
        
        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        generate_ids = self.model.generate(
            **inputs, 
            max_length=256, 
            use_cache=False,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("response", str(response))
        return response

    def process_frame(self, frame_bytes: bytes) -> tuple[bool, torch.Tensor | None]:
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

    def save_buffer_to_mp3(self, waveform: torch.Tensor, sample_rate: int) -> str:

        if waveform.shape[0] > 1:
            waveform = waveform[0].unsqueeze(0)

        # Convert to NumPy int16 (pydub expects this)
        samples = (waveform.squeeze().numpy() * 32767).astype(np.int16)

        # Convert to AudioSegment
        audio_segment = AudioSegment(
            samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit => 2 bytes
            channels=1
        )

        filename = f"{uuid.uuid4()}.mp3"
        audio_segment.export(filename, format="mp3")
        return filename
