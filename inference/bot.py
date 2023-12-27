import os
import sys
from flask import Flask, request, jsonify
from pyngrok import ngrok

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: PYTHONPATH hacks are never a good idea. clean this up later
sys.path.append(os.path.join(INFERENCE_DIR, '..'))

import torch
import argparse
import conversation as convo
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList
from accelerate import infer_auto_device_map, init_empty_weights

# Define the Flask app
app = Flask(__name__)

# Set up ngrok
ngrok_tunnel = ngrok.connect(5000)
print(" * ngrok URL: " + str(ngrok_tunnel.public_url) + " -> http://127.0.0.1:5000/")

class StopWordsCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_words, stream_callback):
        self._tokenizer = tokenizer
        self._stop_words = stop_words
        self._partial_result = ''
        self._stream_buffer = ''
        self._stream_callback = stream_callback

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        first = not self._partial_result
        text = self._tokenizer.decode(input_ids[0, -1])
        self._partial_result += text
        for stop_word in self._stop_words:
            if stop_word in self._partial_result:
                return True
        if self._stream_callback:
            if first:
                text = text.lstrip()
            for stop_word in self._stop_words:
                for i in range(1, len(stop_word)):
                    if self._partial_result.endswith(stop_word[0:i]):
                        self._stream_buffer += text
                        return False
            self._stream_callback(self._stream_buffer + text)
            self._stream_buffer = ''
        return False

class ChatModel:
    human_id = "<human>"
    bot_id = "<bot>"

    def __init__(self, model_name, gpu_id, max_memory):
        device = torch.device('cuda', gpu_id)

        if max_memory is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto")
            self._model.to(device)
        else:
            config = AutoConfig.from_pretrained(model_name)
            with init_empty_weights():
                model_from_conf = AutoModelForCausalLM.from_config(config)

            model_from_conf.tie_weights()

            device_map = infer_auto_device_map(
                model_from_conf,
                max_memory=max_memory,
                no_split_module_classes=["GPTNeoXLayer"],
                dtype="float16"
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                offload_folder="offload",
                offload_state_dict=True,
                torch_dtype=torch.float16
            )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def do_inference(self, prompt, max_new_tokens, do_sample, temperature, top_k, stream_callback=None):
        stop_criteria = StopWordsCriteria(self._tokenizer, [self.human_id], stream_callback)
        inputs = (
            self._tokenizer(prompt, return_tensors='pt')
            .to(self._model.device)
        )
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self._tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stop_criteria]),
        )
        output = self._tokenizer.batch_decode(outputs)[0]

        output = output[len(prompt):]

        return output

# ... (Rest of the code for OpenChatKitShell class and main() function)

if __name__ == '__main__':
    main()