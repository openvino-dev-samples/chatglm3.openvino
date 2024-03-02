# chatglm3.openvino

This is a sample for how to deploy ChatGLM3 with OpenVINO

**1. Set-up the environments:**

```
python3 -m venv openvino_env

source openvino_env/bin/activate

python3 -m pip install --upgrade pip

pip install wheel setuptools

pip install -r requirements.txt
```

**2. Download and Convert model:**

```
python3 convert.py
```
Paramters:
* `--model_id` - model_id for downloading from huggngface_hub (https://huggingface.co/models) or path with directory where pytorch model located.
*  `--output` - path to save model.
* For PRC user, you can download model with `mirror-hf` in advance
    Linux
    ```
    export HF_ENDPOINT=https://hf-mirror.com
    ```
    Windows Powershell
    ```
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    ```
    Download model
    ```
    huggingface-cli download --resume-download --local-dir-use-symlinks False THUDM/chatglm3-6b --local-dir chatglm3-6b
    ```


**3. Quantize model:**

```
python3 quantize.py
```

Paramters:
* `--model_path` - path with directory where OpenVINO IR model located.
* `--precision` - quantization precison: int8 or int4.
* `--output` - path to save model.

**4. Run streaming chatbot:**

```
python3 chat.py -m "./chatglm3_compressed"
```

Paramters:
* `--model_path` - path with directory where OpenVINO IR model located.
* `--max_sequence_length` - maximun size of output tokens.
* `--device` - device to run inference.

Examples:
```
用户: 你好
AI助手: 你好！有什么我可以帮助你的吗？

用户: 你是谁？     
AI助手: 我是一个名为ChatGLM3-6B的人工智能助手，是由清华大学KEG实验室和智谱AI 公司于2023 年共同训练的语言模型开发而成。我的任务是针对用户的问题和要求提供适当的答复和支持。

用户: 请给我讲一个有趣的故事
AI助手: 从前，有一个名叫小明的小男孩，他是一个非常喜欢动物的人。有一天，他在森林里散步时，发现了一个非常漂亮的小鸟。小鸟受伤了，无法飞行。小明非常心疼，于是决定照顾这只小鸟。小明带着小鸟回家，为它搭建了一个小小的巢穴，并找来了一些软草和食物。每天，他都会给小鸟喂食，并为它换水。渐渐地，小鸟的伤势好了起来，开始在小明的家里飞来飞去，它们成了非常好的朋友。然而，一天，小明的父母告诉他，他们必须把小明养的小鸟送到森林里去。小明非常伤心，因为他已经和小鸟成为了好朋友。但是，他的父母告诉他，小鸟在森林里会更加自由自在，而且他也可以继续观看小鸟在森林中的生活。于是，小明和他的父母一起将小鸟送到了森林中。小鸟非常高兴，因为它又可以飞行了，并且还有许多其他的小动物朋友。小明也感到非常开心，因为他知道，即使不能一直拥有小鸟，他仍然可以欣赏到它们在自然中的美丽。从此以后，小明常常来到森林中，寻找小鸟。

用户: 请给这个故事起一个标题
AI助手: 《友谊的力量：小明与小鸟的森林冒险》
```