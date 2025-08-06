from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc


class QwenLLM:
    def __init__(self, model_name_or_path="Qwen/Qwen3-4B", device=None, pipe=None, tokenizer=None):
        # Kiểm tra RAM khả dụng
        print("load_check_point", model_name_or_path)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if pipe is not None and tokenizer is not None:
            self.model = pipe
            self.tokenizer = tokenizer
        else:

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16

                print("Using CUDA.")

                # load the tokenizer and the model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            else:
                print("Using CPU.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    device_map="cpu",
                )
        # self.model = self.model.to(self.device)
        # self.model.eval()

    def generate(self, prompt, max_new_tokens=100, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"result: {result}")
        return result[len(prompt):].strip() if result.startswith(prompt) else result.strip() 