import argparse
from typing import List, Tuple
from threading import Thread
import torch
from optimum.intel.openvino import OVModelForCausalLM
from transformers import (AutoTokenizer, AutoConfig,
                          TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


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
    parser.add_argument('-l',
                        '--max_sequence_length',
                        default=256,
                        required=False,
                        type=int,
                        help='Maximun length of output')
    parser.add_argument('-d',
                        '--device',
                        default='CPU',
                        required=False,
                        type=str,
                        help='Device for inference')
    parser.add_argument('-g',
                        '--gradio',
                        default=False,
                        required=False,
                        type=bool,
                        help='Whether show Gradio interface')
    args = parser.parse_args()
    model_dir = args.model_path

    ov_config = {"PERFORMANCE_HINT": "LATENCY",
                 "NUM_STREAMS": "1", "CACHE_DIR": ""}

    def convert_history_to_token(history: List[Tuple[str, str]]):
        messages = []
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})

        model_inputs = tokenizer.apply_chat_template(messages,
                                                     add_generation_prompt=True,
                                                     tokenize=True,
                                                     return_tensors="pt")
        return model_inputs

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)

    print("====Compiling model====")
    ov_model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=args.device,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )

    stop_tokens = [0, 2]
    stop_tokens = [StopOnTokens(stop_tokens)]

    if not args.gradio:
        history = []
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )
        print("====Starting conversation====")
        while True:
            input_text = input("用户: ")
            if input_text.lower() == 'stop':
                break

            if input_text.lower() == 'clear':
                history = []
                print("AI助手: 对话历史已清空")
                continue

            print("ChatGLM3-6B-OpenVINO:", end=" ")
            history = history + [[parse_text(input_text), ""]]
            model_inputs = convert_history_to_token(history)
            generate_kwargs = dict(
                input_ids=model_inputs,
                max_new_tokens=args.max_sequence_length,
                temperature=0.1,
                do_sample=True,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.1,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList(stop_tokens)
            )

            t1 = Thread(target=ov_model.generate, kwargs=generate_kwargs)
            t1.start()

            partial_text = ""
            for new_text in streamer:
                new_text = new_text
                print(new_text, end="", flush=True)
                partial_text += new_text
            print("\n")
            history[-1][1] = partial_text
    else:
        import gradio as gr

        def bot(history, temperature, top_p, top_k, repetition_penalty):
            model_inputs = convert_history_to_token(history)
            streamer = TextIteratorStreamer(
                tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
            generate_kwargs = dict(
                input_ids=model_inputs,
                max_new_tokens=args.max_sequence_length,
                temperature=temperature,
                do_sample=temperature > 0.0,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList(stop_tokens)
            )
            t = Thread(target=ov_model.generate, kwargs=generate_kwargs)
            t.start()

            for new_token in streamer:
                if new_token != '':
                    history[-1][1] += new_token
                    yield history

        def user(query, history):
            return "", history + [[parse_text(query), ""]]

        chinese_examples = [
            ["你好!"],
            ["你是谁?"],
            ["请介绍一下上海"],
            ["请介绍一下英特尔公司"],
            ["晚上睡不着怎么办？"],
            ["给我讲一个年轻人奋斗创业最终取得成功的故事。"],
            ["给这个故事起一个标题。"],
        ]

        def request_cancel():
            ov_model.request.cancel()

        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=".disclaimer {font-variant-caps: all-small-caps;}",
        ) as demo:
            gr.Markdown(
                f"""<h1><center>OpenVINO ChatGLM3-6b Chatbot</center></h1>""")
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                with gr.Column():
                    msg = gr.Textbox(
                        label="Chat Message Box",
                        placeholder="Chat Message Box",
                        show_label=False,
                        container=False,
                    )
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button("Submit")
                        stop = gr.Button("Stop")
                        clear = gr.Button("Clear")
            with gr.Row():
                with gr.Accordion("Advanced Options:", open=False):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Temperature",
                                    value=0.1,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.1,
                                    interactive=True,
                                    info="Higher values produce more diverse outputs",
                                )
                        with gr.Column():
                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-p (nucleus sampling)",
                                    value=1.0,
                                    minimum=0.0,
                                    maximum=1,
                                    step=0.01,
                                    interactive=True,
                                    info=(
                                        "Sample from the smallest possible set of tokens whose cumulative probability "
                                        "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                    ),
                                )
                        with gr.Column():
                            with gr.Row():
                                top_k = gr.Slider(
                                    label="Top-k",
                                    value=50,
                                    minimum=0.0,
                                    maximum=200,
                                    step=1,
                                    interactive=True,
                                    info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                                )
                        with gr.Column():
                            with gr.Row():
                                repetition_penalty = gr.Slider(
                                    label="Repetition Penalty",
                                    value=1.1,
                                    minimum=1.0,
                                    maximum=2.0,
                                    step=0.1,
                                    interactive=True,
                                    info="Penalize repetition — 1.0 to disable.",
                                )
            gr.Examples(
                chinese_examples, inputs=msg, label="Click on any example and press the 'Submit' button"
            )

            submit_event = msg.submit(
                fn=user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).then(
                fn=bot,
                inputs=[
                    chatbot,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty
                ],
                outputs=chatbot,
                queue=True,
            )
            submit_click_event = submit.click(
                fn=user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).then(
                fn=bot,
                inputs=[
                    chatbot,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty
                ],
                outputs=chatbot,
                queue=True,
            )
            stop.click(
                fn=request_cancel,
                inputs=None,
                outputs=None,
                cancels=[submit_event, submit_click_event],
                queue=False,
            )
            clear.click(lambda: None, None, chatbot, queue=False)

        # if you are launching remotely, specify server_name and server_port
        #  demo.launch(server_name='your server name', server_port='server port in int')
        # if you have any issue to launch on your platform, you can pass share=True to launch method:
        # demo.launch(share=True)
        # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
        demo.launch()
