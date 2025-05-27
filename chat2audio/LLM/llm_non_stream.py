# llm_stream.py
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM, TextIteratorStreamer
import re
from threading import Thread
import asyncio
from typing import AsyncGenerator

from LLM.prompts import speaker_prompts

class LLMNonStreamer:
    def __init__(self, model_path="/root/autodl-tmp/Qwen2.5-7B"):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="cuda:0", 
            trust_remote_code=True
        ).eval()

    def chat(self, user_input : str, speaker : str, temperature : float = 0.7):
        message = [
            {"role": "system", "content": speaker_prompts[speaker]},
            {"role": "user", "content": user_input}
        ]
        text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=1024, temperature=temperature)
        generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pattern = r'\{[^{}]*\}|\[[^][]*\]|\([^()]*\)|（[^（）]*）|【[^【】]*】|「[^「」]*」|『[^『』]*』|《[^《》]*》|“[^“”]*”'
        return re.sub(pattern, '', response)
