简体中文 | [English](README.md)

# chatglm3.openvino Demo

这是如何使用 OpenVINO 部署 ChatGLM3 的示例

## 1. 环境配置

我们推荐您新建一个虚拟环境，然后按照以下安装依赖。
推荐在python3.10以上的环境下运行该示例。

Linux

```
python3 -m venv openvino_env

source openvino_env/bin/activate

python3 -m pip install --upgrade pip

pip install wheel setuptools

pip install -r requirements.txt
```

Windows Powershell

```
python3 -m venv openvino_env

.\openvino_env\Scripts\activate

python3 -m pip install --upgrade pip

pip install wheel setuptools

pip install -r requirements.txt
```

## 2. 转换模型

由于需要将Huggingface模型转换为OpenVINO IR模型，因此您需要下载模型并转换。

```
python3 convert.py --model_id THUDM/chatglm3-6b --precision int4 --output {your_path}/chatglm3-6b-ov 
```

### 可以选择的参数

* `--model_id` - 用于从 Huggngface_hub (https://huggingface.co/models) 或 模型所在目录的路径（绝对路径）
* `--precision` - 模型精度：fp16, int8 或 int4。
* `--output` - 转换后模型保存的地址
* 如果您访问`huggingface` 有困难，你可以尝试使用 `mirror-hf` 进行下载

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
    huggingface-cli download --resume-download --local-dir-use-symlinks False THUDM/chatglm3-6b --local-dir {your_path}/chatglm3-6b 
    ```

## 3. 运行流式聊天机器人

```
python3 chat.py --model_path {your_path}/chatglm3-6b-ov --max_sequence_length 4096 --device CPU
```

### 可以选择的参数

* `--model_path` - OpenVINO IR 模型所在目录的路径。
* `--max_sequence_length` - 输出标记的最大大小。
* `--device` - 运行推理的设备。例如："CPU","GPU"。

## 例子

```
用户: 你好
ChatGLM3-6B-OpenVINO: 你好！有什么我可以帮助你的吗？

用户: 你是谁？     
ChatGLM3-6B-OpenVINO: 我是一个名为ChatGLM3-6B的人工智能助手，是由清华大学KEG实验室和智谱AI 公司于2023 年共同训练的语言模型开发而成。我的任务是针对用户的问题和要求提供适当的答复和支持。

用户: 请给我讲一个有趣的故事
ChatGLM3-6B-OpenVINO: 从前，有一个名叫小明的小男孩，他是一个非常喜欢动物的人。有一天，他在森林里散步时，发现了一个非常漂亮的小鸟。小鸟受伤了，无法飞行。小明非常心疼，于是决定照顾这只小鸟。小明带着小鸟回家，为它搭建了一个小小的巢穴，并找来了一些软草和食物。每天，他都会给小鸟喂食，并为它换水。渐渐地，小鸟的伤势好了起来，开始在小明的家里飞来飞去，它们成了非常好的朋友。然而，一天，小明的父母告诉他，他们必须把小明养的小鸟送到森林里去。小明非常伤心，因为他已经和小鸟成为了好朋友。但是，他的父母告诉他，小鸟在森林里会更加自由自在，而且他也可以继续观看小鸟在森林中的生活。于是，小明和他的父母一起将小鸟送到了森林中。小鸟非常高兴，因为它又可以飞行了，并且还有许多其他的小动物朋友。小明也感到非常开心，因为他知道，即使不能一直拥有小鸟，他仍然可以欣赏到它们在自然中的美丽。从此以后，小明常常来到森林中，寻找小鸟。

用户: 请给这个故事起一个标题
ChatGLM3-6B-OpenVINO: 《友谊的力量：小明与小鸟的森林冒险》
```

## 常见问题
1. 为什么倒入本地模型还会报 huggingface 链接错误
   - 降级 transformers 库到 4.37.2 版本

2. 需要安装 OpenVINO C++ 推理引擎吗
   - 不需要

3. 一定要使用 Intel 的硬件吗？
   - 我们仅在 Intel 设备上尝试，我们推荐使用x86架构的英特尔设备，包括但不限制于：
   - 英特尔的CPU，包括个人电脑CPU 和服务器CPU。
   - 英特尔的集成显卡。 例如：Arc™，Iris® 系列。
   - 英特尔的独立显卡。例如：ARC™ A770 显卡。
  
4. 为什么OpenVINO没检测到我系统上的GPU设备？
   - 确保OpenCL驱动是安装正确的。
   - 确保你有足够的权限访问GPU设备
   - 更多信息可以参考[Install GPU drivers](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu#1-install-python-git-and-gpu-drivers-optional)

5. 是否支持C++？
   - C++示例可以[参考](https://github.com/openvinotoolkit/openvino.genai/tree/master/text_generation/causal_lm/cpp)
