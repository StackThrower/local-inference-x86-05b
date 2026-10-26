import os
import threading
from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = os.getenv("MODEL_PATH", "Qwen/Qwen3.5-0.8B")
NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "0"))


class ChatModel:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        if NUM_THREADS > 0:
            torch.set_num_threads(NUM_THREADS)
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.eval()
        # CPU generation is compute-bound; running requests in parallel only slows each of them down.
        self._lock = threading.Lock()

    def _prepare(self, messages: list[dict], max_new_tokens: int, temperature: float, top_p: float) -> dict:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        kwargs = dict(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        return kwargs

    @torch.inference_mode()
    def generate(self, messages: list[dict], max_new_tokens: int, temperature: float, top_p: float) -> dict:
        kwargs = self._prepare(messages, max_new_tokens, temperature, top_p)
        prompt_len = kwargs["input_ids"].shape[1]
        with self._lock:
            output = self.model.generate(**kwargs)
        new_tokens = output[0][prompt_len:]
        return {
            "content": self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            "prompt_tokens": prompt_len,
            "completion_tokens": len(new_tokens),
        }

    def stream(self, messages: list[dict], max_new_tokens: int, temperature: float, top_p: float) -> Iterator[str]:
        kwargs = self._prepare(messages, max_new_tokens, temperature, top_p)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run() -> None:
            with self._lock, torch.inference_mode():
                self.model.generate(**kwargs, streamer=streamer)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()
