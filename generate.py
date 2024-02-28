import argparse
from optimum.intel.openvino import OVModelForCausalLM
from transformers import (PretrainedConfig, AutoTokenizer, AutoConfig,
                          TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)

from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino.utils import OV_XML_FILE_NAME
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path
from threading import Thread
import torch


def text_processor(new_text):
    new_text = new_text.strip()
    new_text = new_text.replace("[[训练时间]]", "2023年")
    return new_text


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class OVCHATGLMModel(OVModelForCausalLM):
    """
    Optimum intel compatible model wrapper for CHATGLM2
    """

    def __init__(
        self,
        model: "Model",
        config: "PretrainedConfig" = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf["chatglm"] = NormalizedTextConfig.with_args(
            num_layers="num_hidden_layers",
            num_attention_heads="num_attention_heads",
            hidden_size="hidden_size",
        )
        super().__init__(
            model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs
        )

    def _reshape(
        self,
        model: "Model",
        *args, **kwargs
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            input_name = inputs.get_any_name()
            if input_name.startswith('beam_idx'):
                continue
            if input_name.startswith('past_key_values'):
                shapes[inputs][1] = -1
                shapes[inputs][2] = 2
            elif shapes[inputs].rank.get_length() > 1:
                shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path)
        init_cls = OVCHATGLMModel

        return init_cls(
            model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_path',
                        required=True,
                        type=str,
                        help='Required. model path')
    parser.add_argument('-p',
                        '--prompt',
                        default="请介绍一下上海？",
                        required=False,
                        type=str,
                        help='Required. prompt sentence')
    parser.add_argument('-l',
                        '--max_sequence_length',
                        default=128,
                        required=False,
                        type=int,
                        help='Required. maximun lengh of output')
    parser.add_argument('-d',
                        '--device',
                        default='CPU',
                        required=False,
                        type=str,
                        help='Required. device for inference')
    parser.add_argument('-c',
                        '--chat',
                        default=False,
                        required=False,
                        type=bool,
                        help='generation or chat mode')
    args = parser.parse_args()

    start_message = "你是一个乐于助人、尊重他人以及诚实可靠的助手。在安全的情况下，始终尽可能有帮助地回答。 您的回答不应包含任何有害、不道德、种族主义、性别歧视、有毒、危险或非法的内容。请确保您的回答在社会上是公正的和积极的。如果一个问题没有任何意义或与事实不符，请解释原因，而不是回答错误的问题。如果您不知道问题的答案，请不要分享虚假信息。另外，答案请使用中文。"
    ov_config = {"PERFORMANCE_HINT": "LATENCY",
                 "NUM_STREAMS": "1", "CACHE_DIR": ""}

    tok = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    model_dir = args.model_path

    ov_model = OVCHATGLMModel.from_pretrained(
        model_dir,
        device=args.device,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )

    streamer = TextIteratorStreamer(
        tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    stop_tokens = [2, 64795, 64797]
    stop_tokens = [StopOnTokens(stop_tokens)]

    def convert_history_to_token(history: List[Tuple[str, str]]):
        input_ids = []
        input_ids.extend(tok.build_single_message("system", "", start_message))
        for old_query, response in history[:-1]:
            input_ids.extend(tok.build_single_message("user", "", old_query))
            input_ids.extend(tok.build_single_message(
                "assistant", "", response))
        input_ids.extend(tok.build_single_message(
            "user", "", history[-1][0]))
        input_ids.extend([tok.get_command("<|assistant|>")])
        input_token = tok.batch_encode_plus(
            [input_ids], return_tensors="pt", is_split_into_words=True
        ).input_ids
        return input_token

    history = []
    while True:
        input_text = input("用户: ")
        if input_text.lower() == 'stop':
            break

        if input_text.lower() == 'clear':
            history = []
            print("ChatGLM3-6B: 对话历史已清空")
            continue
        history = history + [[input_text, ""]]

        input_ids = convert_history_to_token(history)

        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=args.max_sequence_length,
            temperature=0.1,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList(stop_tokens)
        )

        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            ov_model.generate(**generate_kwargs)

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()
        print("ChatGLM3-6B:", end="")
        partial_text = ""
        for new_text in streamer:
            new_text = text_processor(new_text)
            print(new_text, end="", flush=True)
            partial_text += new_text
        print("\n")
        history[-1][1] = partial_text
