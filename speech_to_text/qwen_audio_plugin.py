from .asr_base import AsrBase, register_plugin
import uuid
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from pydub import AudioSegment
from silero_vad import load_silero_vad
from typing import Union
import numpy as np
import base64
import librosa

@register_plugin("qwen_audio")
class QwenAudioPlugin(AsrBase):
    def __init__(self, model_id: str = "base", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.language = kwargs.get("language", None)
        self.model = None
        self.processor = None
        self.SAMPLE_RATE = 16000  # Qwen expects 16kHz audio

    def load(self):
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",                                              
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
        )
        return self
    
    def predict(self, audio: Union[str, np.ndarray, bytes]) -> str:
        if self.model is None or self.processor is None:
            self.load()

        # Handle file path, numpy array, or base64 bytes
        if isinstance(audio, str):
            if audio.startswith('data:') or len(audio) > 255:
                # Base64 string
                audio_bytes = base64.b64decode(audio)
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # File path -> preprocess trả về np.ndarray
                audio_np = self.preprocess(audio)
        elif isinstance(audio, bytes):
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio, np.ndarray):
            audio_np = audio.astype(np.float32, copy=False)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        # Đảm bảo shape là (1, T) cho model
        if audio_np.ndim == 1:
            audio_np = np.expand_dims(audio_np, axis=0)  # (1, T)

        # Convert sang torch.Tensor
        final_waveform = torch.from_numpy(audio_np)
        path = self.save_buffer_to_mp3(final_waveform, self.SAMPLE_RATE)

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a speech recognition assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": path},
                    {"type": "text", "text": "What does the person say?"},
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
        
        inputs = self.processor(text=text, audio=audios, return_tensors="pt", padding=True)
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
