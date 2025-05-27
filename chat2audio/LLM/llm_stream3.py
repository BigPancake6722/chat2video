import re
import asyncio
from threading import Thread
from typing import AsyncGenerator, Optional
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

    async def stream_output(self, user_input: str, speaker_id: str, prompt: Optional[str] = None, temperature=1.4) -> AsyncGenerator[str, None]:
        """流式生成已清理括号的文本"""
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        prompt = speaker_prompts.get(speaker_id, prompt) if prompt is None else prompt
        if prompt is None:
            raise ValueError("No prompt provided and speaker_id not found in speaker_prompts")
        
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
        
        try:
            buffer = ""
            for new_text in streamer:
                buffer += new_text
                # 尝试清理缓冲区
                cleaned = self.bracket_pattern.sub('', buffer)
                if cleaned != buffer:  # 如果有括号被移除
                    if cleaned:  # 确保不发送空字符串
                        yield cleaned
                    buffer = ""
                elif len(buffer) > 100:  # 防止缓冲区过大
                    yield buffer
                    buffer = ""
            
            # 处理最后剩余的缓冲区内容
            if buffer:
                yield self.bracket_pattern.sub('', buffer)
        finally:
            thread.join()