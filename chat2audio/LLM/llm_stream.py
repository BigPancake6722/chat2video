import re
import asyncio
from threading import Thread
from typing import AsyncGenerator
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM, TextIteratorStreamer

from LLM.prompts import speaker_prompts

class LLMStreamer:
    def __init__(self, model_path="/root/autodl-tmp/Qwen2.5-7B"):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="cuda:0", 
            trust_remote_code=True
        ).eval()
        self.bracket_pattern = re.compile(
            r'\{[^{}]*\}|\[[^][]*\]|\([^()]*\)|（[^（）]*）|【[^【】]*】|「[^「」]*」|『[^『』]*』|《[^《》]*》|“[^“”]*”'
        )

    async def stream_output(self, user_input: str, speaker_id: str, prompt : str, temperature=0.7) -> AsyncGenerator[str, None]:
        """流式生成已清理括号的文本"""
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        prompt = speaker_prompts[speaker_id] if prompt == None else prompt
        
        message = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
        text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=temperature,
            do_sample=temperature > 0
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        buffer = ""
        for new_text in streamer:
            buffer += new_text
            cleaned = self.bracket_pattern.sub('', buffer)
            if cleaned:
                yield cleaned
                buffer = ""
        
        thread.join()