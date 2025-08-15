from .asr_base import AsrBase, register_plugin
import uuid
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
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

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(self.model_id, device_map="auto")
        return self
    
    def predict(self, audio: Union[str, np.ndarray, bytes]) -> str:
        if self.model is None or self.processor is None:
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

        audio_tensors = [torch.tensor(chunk, dtype=torch.float32) for chunk in audio_data]
        final_waveform = torch.cat(audio_tensors, dim=1)
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
