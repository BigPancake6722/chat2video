import os
import sys
import wave
import signal
import argparse
import traceback
import subprocess
import numpy as np
import soundfile as sf
from io import BytesIO
from typing import Generator
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File

sys.path.append('/root/chat2video/chat2audio/tts/')
sys.path.append('/root/chat2video/chat2audio/tts/GPT_SoVITS')
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names


class TTSNonStreamer:
    def __init__(self):
        i18n = I18nAuto()
        self.cut_method_names = get_cut_method_names()

        parser = argparse.ArgumentParser(description="GPT-SoVITS api")
        parser.add_argument("-c", "--tts_config", type=str, default="/root/chat2video/chat2audio/tts/GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
        parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
        parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
        args = parser.parse_args()
        config_path = args.tts_config
        # device = args.device
        port = args.port
        host = args.bind_addr
        argv = sys.argv

        if config_path in [None, ""]:
            config_path = "/root/chat2video/chat2audio/tts/GPT_SoVITS/configs/tts_infer.yaml"

        self.tts_config = TTS_Config(config_path)
        print(self.tts_config)
        self.tts_pipeline = TTS(self.tts_config)

    # process the output audio type
    def pack_ogg(self, io_buffer:BytesIO, data:np.ndarray, rate:int):
        with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
            audio_file.write(data)
        return io_buffer

    def pack_raw(self, io_buffer:BytesIO, data:np.ndarray, rate:int):
        io_buffer.write(data.tobytes())
        return io_buffer

    def pack_wav(self, io_buffer:BytesIO, data:np.ndarray, rate:int):
        io_buffer = BytesIO()
        sf.write(io_buffer, data, rate, format='wav')
        return io_buffer

    def pack_aac(self, io_buffer:BytesIO, data:np.ndarray, rate:int):
        process = subprocess.Popen([
            'ffmpeg',
            '-f', 's16le',  # 输入16位有符号小端整数PCM
            '-ar', str(rate),  # 设置采样率
            '-ac', '1',  # 单声道
            '-i', 'pipe:0',  # 从管道读取输入
            '-c:a', 'aac',  # 音频编码器为AAC
            '-b:a', '192k',  # 比特率
            '-vn',  # 不包含视频
            '-f', 'adts',  # 输出AAC数据流格式
            'pipe:1'  # 将输出写入管道
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate(input=data.tobytes())
        io_buffer.write(out)
        return io_buffer

    def pack_audio(self, io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
        if media_type == "ogg":
            io_buffer = self.pack_ogg(io_buffer, data, rate)
        elif media_type == "aac":
            io_buffer = self.pack_aac(io_buffer, data, rate)
        elif media_type == "wav":
            io_buffer = self.pack_wav(io_buffer, data, rate)
        else:
            io_buffer = self.pack_raw(io_buffer, data, rate)
        io_buffer.seek(0)
        return io_buffer

    # create the audio from text on specific speaker
    def check_params(self, req:dict):
        text:str = req.get("text", "")
        text_lang:str = req.get("text_lang", "")
        ref_audio_path:str = req.get("ref_audio_path", "")
        streaming_mode:bool = req.get("streaming_mode", False)
        media_type:str = req.get("media_type", "wav")
        prompt_lang:str = req.get("prompt_lang", "")
        text_split_method:str = req.get("text_split_method", "cut5")

        if ref_audio_path in [None, ""]:
            print("ref_audio_path is required")
            return False
        if text in [None, ""]:
            print("text is required")
            return False
        
        if (text_lang in [None, ""]) :
            print("text_lang is required")
            return False
        elif text_lang.lower() not in self.tts_config.languages:
            print(f"text_lang: {text_lang} is not supported in version {self.tts_config.version}")
            return False
        
        if (prompt_lang in [None, ""]) :
            print("prompt_lang is required")
            return False
        elif prompt_lang.lower() not in self.tts_config.languages:
            print(f"prompt_lang: {prompt_lang} is not supported in version {self.tts_config.version}")
            return False
        
        if media_type not in ["wav", "raw", "ogg", "aac"]:
            print(f"media_type: {media_type} is not supported")
            return False
        elif media_type == "ogg" and  not streaming_mode:
            print("ogg format is not supported in non-streaming mode")
            return False

        if text_split_method not in self.cut_method_names:
            print(f"text_split_method:{text_split_method} is not supported")
            return False

        return True

    def wave_header_chunk(self, frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
        wav_buf = BytesIO()
        with wave.open(wav_buf, "wb") as vfout:
            vfout.setnchannels(channels)
            vfout.setsampwidth(sample_width)
            vfout.setframerate(sample_rate)
            vfout.writeframes(frame_input)

        wav_buf.seek(0)
        return wav_buf.read()

    async def tts_handle(self, req:dict):
        """
        Text to speech handler.
        
        Args:
            req (dict): 
                {
                    "text": "",                   # str.(required) text to be synthesized
                    "text_lang: "",               # str.(required) language of the text to be synthesized
                    "ref_audio_path": "",         # str.(required) reference audio path
                    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                    "prompt_text": "",            # str.(optional) prompt text for the reference audio
                    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                    "top_k": 5,                   # int. top k sampling
                    "top_p": 1,                   # float. top p sampling
                    "temperature": 1,             # float. temperature for sampling
                    "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                    "batch_size": 1,              # int. batch size for inference
                    "batch_threshold": 0.75,      # float. threshold for batch splitting.
                    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                    "seed": -1,                   # int. random seed for reproducibility.
                    "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                    "streaming_mode": False,      # bool. whether to return a streaming response.
                    "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                    "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.          
                }
        returns:
            StreamingResponse: audio stream response.
        """
        
        streaming_mode = req.get("streaming_mode", False)
        return_fragment = req.get("return_fragment", False)
        media_type = req.get("media_type", "wav")

        check_res = self.check_params(req)
        if not check_res:
            return None

        if streaming_mode or return_fragment:
            req["return_fragment"] = True
        
        try:
            tts_generator=self.tts_pipeline.run(req)
            
            if streaming_mode:
                def streaming_generator(tts_generator:Generator, media_type:str):
                    if media_type == "wav":
                        yield wave_header_chunk()
                        media_type = "raw"
                    for sr, chunk in tts_generator:
                        yield self.pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
                return streaming_generator(tts_generator, media_type, )
        
            else:
                sr, audio_data = next(tts_generator)
                audio_data = self.pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
                return audio_data
        except Exception as e:
            print("tts failed, Exception", str(e))
            return None


    async def tts_get_endpoint(self, 
        text,
        speaker_id : str = "Firefly", 
        media_type : str = "wav"):
        from speaker_params import speakers
        speaker = speakers[speaker_id]
        self.tts_pipeline.init_vits_weights(speaker["sovits_model"])
        self.tts_pipeline.init_t2s_weights(speaker["gpt_model"])
        req = {
            "text" : text,
            "text_lang" : speaker["target_language"],
            "ref_audio_path" : speaker["ref_audio"],
            "aux_ref_audio_paths" : None,
            "prompt_text" : speaker["ref_text"],
            "prompt_lang" : speaker["ref_language"],
            "top_k" : speaker['top_k'],
            "top_p" : speaker['top_p'],
            "temperature" : speaker['temperature'],
            "text_split_method" : "cut0",
            "batch_size" : int(1),
            "batch_threshold" : float(0.75),
            "speed_factor" : speaker["speed_factor"],
            "split_bucket" : True,
            "fragment_interval" : float(0.3),
            "seed" : int(-1),
            "media_type" : media_type,
            "streaming_mode" : False,
            "parallel_infer" : True,
            "repetition_penalty" : float(1.35)
        }
        return await self.tts_handle(req)

# if __name__ == "__main__":
#     audio = tts_get_endpoint("我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可")
#     save_wav("./output.wav", audio, 32000)
    