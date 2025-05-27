# chat2audio

## some notes

- 模型为基于[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)与[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)构建的文本-语音问答模型，在文本转语音时实现了本地化推理的过程，不需要通过4次调用。
- 模型中使用的Qwen2.5-7B模型经过[Chinese-Roleplay-SingleTurn](https://huggingface.co/datasets/LooksJuicy/Chinese-Roleplay-SingleTurn)数据集以及部分游戏中的语音对话使用[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)进行微调以适应使用不同的prompt实现不同角色扮演的需求。微调的完整数据集可以从[此处](https://huggingface.co/datasets/BigPancake01/roleplayLLM_Chinese)获取，微调后的模型可以从[此处](https://huggingface.co/BigPancake01/rolePlay_Qwen2.5-7B)获取。
- 为减少内存占用，模型中使用的GPT-SoVITS模型在原有基础上进行了一定的修改，仅包括模型推理部分的代码。项目中需要通过传入由原模型进行训练得到的GPT模型以及VITS模型以及参考音频进行推理。项目中提供了部分训练的模型进行参考，可以在[此处](https://huggingface.co/BigPancake01/GPT-SoVITS_Mihoyo)获取。

**项目中涉及到部分公开的音频采样数据集以及文本数据集，相关内容应遵循开源且不可商用的原则。**

## 项目部分参数说明

项目中为了方便理解与文件读取采用了部分绝对路径的读取方式，以下做出详细说明。

### llm模型路径

- 位置：LLM.llm_non_stream.py LLM.llm_stream.py	model_path 
- 该路径应包含Qwen训练模型中的所有文件

### GPT-SoVITS预训练模型路径

- 位置：tts.config.py	cnhubert_path bert_path pretrained_sovits_path pretrained_gpt_path

  ​	   tts.GPT-SoVITS.config.tts_infer.yaml

- 该路径指向GPT-SoVITS预训练模型文件夹中特定的模型权重文件，需要检查路径是否正确。

### GPT-SoVITS推理参数

- 位置：tts.speaker_params.py
- 该文件中包含以dict表示的不同说话人推理时使用的参数，在进行推理时直接调用以方便更换说话人。

​	下列参数中涉及到路径的，推荐使用绝对路径写入。

| 参数名          | 释义                                                         |
| --------------- | ------------------------------------------------------------ |
| gpt_model       | GPT模型权重文件路径，需具体到对应的权重文件                  |
| sovits_model    | SoVITS模型权重文件路径，需具体到对应的权重文件               |
| ref_audio       | 参考音频路径，根据源项目要求，长度在3s~10s之间，与推理的说话人相同 |
| ref_text        | 参考音频的文字，即需要给出参考音频中说话人说了什么           |
| ref_language    | 参考音频的语种，需要缩写，如汉语-zh，英语-en                 |
| target_language | 推理输出音频的语种，需要缩写。                               |
| speed_factor    | 推理输出音频的语速，1表示与训练数据集相同的语速。            |
| top_k           | 采样样本数，默认为int(5)                                     |
| top_p           | 采样样本数，默认为float(1.0)                                 |
| temperature     | 推理温度系数，大于1时采样策略更激进。默认为float(1.0)        |

### LLM角色扮演prompt

- 位置：LLM.prompts.py
- 该文件中包含以dict表示的不同角色在进行角色扮演时相应的prompt，目前使用的版本主要基于[豆包](https://www.doubao.com/chat)生成，包含角色的经历，身份以及与对话者的关系的描述。在进行llm调用未指定prompt的情况下，模型根据speaker调用对应角色的prompt文件。

## 项目中需要补全的文件

该模型中包含部分较大的预训练模型，为减少内存占用，github中仅包含主体代码部分，下面给出huggingface中的模型下载与位置。

| 文件属性               | 链接                                                         | 项目中位置                       | 说明                                        |
| ---------------------- | ------------------------------------------------------------ | -------------------------------- | ------------------------------------------- |
| GPT模型权重文件        | [huggingface](https://huggingface.co/BigPancake01/GPT-SoVITS_Mihoyo/tree/main/GPT_models) | tts/GPT_weights_v2               | 选择需要的文件夹下载                        |
| VITS模型权重文件       | [huggingface](https://huggingface.co/BigPancake01/GPT-SoVITS_Mihoyo/tree/main/VITS_models) | tts/SoVITS_weights_v2            | 选择需要的文件夹下载                        |
| GPT-SoVITS预训练模型   | [原项目地址](https://huggingface.co/lj1995/GPT-SoVITS)       | tts/GPT_SoVITS/pretrained_models | 需要将所有文件下载                          |
| GPT-SoVITS中文TTS支持  | [原项目地址](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip) | tts/GPT_SoVITS/text              | 解压重命名为G2PWModel后放置                 |
| Qwen训练模型           | [huggingface](https://huggingface.co/BigPancake01/rolePlay_Qwen2.5-7B) | 空闲位置                         | 需要将所有文件下载                          |
| GPT-SoVITS推理参考音频 | [huggingface](https://huggingface.co/BigPancake01/GPT-SoVITS_Mihoyo/tree/main/ref_audios) | 空闲位置                         | 选择需要的文件下载                          |
| GPT-SoVITS推理参数     | 项目内修改                                                   | tts/config.py                    | 需要根据[提示](#GPT-SoVITS推理参数)进行修改 |
| LLM角色扮演prompt      | 项目内修改                                                   | LLM/prompts.py                   | 需要根据[提示](#LLM角色扮演prompt)进行修改  |

## 项目环境配置及使用

- 模型文件下载

  由于项目中模型文件大部分存储在huggingface平台中，故而对于大模型文件其需要使用git lfs进行下载。

  同时如果在Linux系统重运行时，若文件过大，可以采用`ln -s`生成软链接的进行文件下载。

- 环境配置

  推荐使用conda或minconda进行环境配置。新建conda虚拟环境后，在项目根目录下执行：

  ```shell
  pip install -r requirements.txt
  ```

- 项目运行

  项目主要通过http协议进行请求与响应，同时提供了非流式的音频响应方式以及基于句子分割模式的流式相应方式。

  - 非流式方式

    1. 在启动虚拟环境的条件下在根目录下运行

       ```shell
        python app_non_stream.py
       ```

    ​	若项目成功启动，其将运行在`127.0.0.1:8000`端口。

    2. 确认项目成功启动后，通过post方式请求`http://127.0.0.1:8000/chat2audio`推理音频，请求体如下

       ```json
       {
           "input" : "做个自我介绍吧",
           "speaker" : "Nahida"
       }
       ```

       其中`input`为对大模型提问的内容，`speaker`为说话人id。

    3. 项目成功响应将在响应体中包含一段音频。

  - 流式方式

    1. 在启动虚拟环境的条件下在根目录下运行

       ```shell
       python app_stream.py
       ```

    ​	若项目成功启动，其将运行在`127.0.0.1:8000`端口。

    2. 确认项目成功启动后，通过post方式请求`http://127.0.0.1:8000/chat2audio`推理音频，请求体如下

       ```json
       {
           "input" : "做个自我介绍吧",
           "speaker" : "Nahida"
       }
       ```

       其中`input`为对大模型提问的内容，`speaker`为说话人id。

    3. 项目成功响应将在响应体中包含一段音频。


  上述`speaker`参数需要在GPT-SoVITS推理参数中的key对应，否则无法成功进行推理。