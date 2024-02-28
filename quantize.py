import os
import openvino as ov
from transformers import AutoConfig, AutoTokenizer
import nncf
from pathlib import Path
import argparse
import shutil


def is_gptq(config):
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    return quantization_config and quantization_config["quant_method"] == "gptq"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_path',
                        default='./chatglm3_fp16',
                        required=False,
                        type=str,
                        help='orignal model path')
    parser.add_argument('-o',
                        '--output',
                        default='./chatglm3_compressed',
                        required=False,
                        type=str,
                        help='Required. path to save the compressed ir model')
    parser.add_argument('-p',
                        '--precision',
                        required=False,
                        default="int4",
                        type=str,
                        choices=["int8", "int4"],
                        help='int8 or int4')
    args = parser.parse_args()

    compressed_model_path = Path(args.output)
    orignal_model_path = Path(args.model_path)
    if compressed_model_path.exists() == False:
        os.mkdir(compressed_model_path)

    model_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True)
    gptq_applied = is_gptq(model_config)

    print("====loading model====")
    if not orignal_model_path.exists():
        print(" Please run 'export.py' to export IR model to local ")
    else:
        ov_model = ov.Core().read_model(orignal_model_path / "openvino_model.xml")

    if args.precision == "int4" and not gptq_applied:
        print("====exporting int4 model====")
        compressed_model = nncf.compress_weights(
            ov_model, mode=nncf.CompressWeightsMode.INT4_SYM, group_size=128, ratio=0.8)
    elif args.precision == "int8" and not gptq_applied:
        print("====exporting int8 model====")
        compressed_model = nncf.compress_weights(ov_model)
    else:
        raise RuntimeError(
            "Can not quantize a GPTQ model"
        )
    ov.save_model(compressed_model, compressed_model_path /
                  "openvino_model.xml")
    shutil.copy(orignal_model_path / 'config.json',
                compressed_model_path / 'config.json')

    print("====exporting tokenizer====")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(compressed_model_path)
