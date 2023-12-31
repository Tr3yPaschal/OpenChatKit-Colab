#curl -X POST -d "message=Cool nice to meet you" -H "Content-Type: application/x-www-form-urlencoded" -H "Authorization: Bearer your_api_key" https://8bb3bfb341d2.ngrok.app

import os
import sys
from flask import Flask, request, jsonify
from pyngrok import ngrok
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList
from accelerate import infer_auto_device_map, init_empty_weights
#added cors
from flask_cors import CORS
# Define the Flask app
app = Flask(__name__)
CORS(app, origins="*")
api_key = 'a1f7e49d-64df-4f0e-94dd-9629464ed6b9'

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up ngrok
ngrok_tunnel = ngrok.connect(12345)
print(" * ngrok URL: " + str(ngrok_tunnel.public_url) + " -> http://127.0.0.1:12345/")


model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"  # Default model name
max_memory = None  # Default max_memory (can be updated)
gpu_id = 0  # Default GPU ID (can be updated)


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
            # buffer tokens if the partial result ends with a prefix of a stop word, e.g. "<hu"
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
        device = torch.device('cuda', gpu_id) if torch.cuda.is_available() else torch.device('cpu')
        
        # Load the configuration for the model
        config = AutoConfig.from_pretrained(model_name)

        # Initialize the model with empty weights if offloading is needed
        with init_empty_weights():
            self._model = AutoModelForCausalLM.from_config(config)

        # Infer the device map based on available memory
        device_map = infer_auto_device_map(
            self._model, max_memory=max_memory, no_split_module_classes=["GPTNeoXLayer"], dtype="float16"
        )

        # Check if offloading is needed based on the device map
        needs_offloading = any(device == 'disk' for device in device_map.values())

        if needs_offloading:
            # Load the model with device map and offload to disk
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                offload_folder="offload",  # Offload directory
                offload_state_dict=True,
                torch_dtype=torch.float16
            )
            # Offload the model to the disk
            offload_directory = "../offload/"  # Specify the offload directory
            disk_offload(model=self._model, offload_dir=offload_directory)
        else:
            # Load the model normally onto the specified device
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            )
            self._model.to(device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def do_inference(self, prompt, max_new_tokens, do_sample, temperature, top_k, stream_callback=None):
        stop_criteria = StopWordsCriteria(self._tokenizer, [self.human_id], stream_callback)
        inputs = self._tokenizer(
            text=[prompt],  # Make sure to pass the prompt as a list of strings
            return_tensors='pt'
        ).to(self._model.device)
        outputs = self._model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self._tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stop_criteria]),
        )
        output = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # remove the context from the output
        output = output[len(prompt):]

        return output
    
# def check_api_key():
#     if 'Authorization' not in request.headers:
#         return jsonify({'message': 'Missing API key'}), 401

#     provided_key = request.headers['Authorization']

#     if provided_key != f'Bearer {api_key}':
#         return jsonify({'message': 'Invalid API key'}), 401

# # Register the check_api_key function to run before each request
# app.before_request(check_api_key)

@app.route('/', methods=['POST'])
def chat():
    # Get the message from the POST request
    message = request.form.get('message')
    print(message)
    object_type = type(message)
    print(object_type)

    # Perform chat logic
    bot_response = chat_model.do_inference(
        prompt=message,
        max_new_tokens=128,  # Set the maximum number of tokens for the response
        do_sample=True,      # Set to True if you want to sample the response
        temperature=0.6,     # Set the temperature for the LM
        top_k=40,            # Set the top-k value for the LM
        stream_callback=None  # Set a stream_callback if needed
    )

    # Return the chat bot's response as JSON
    response = {"response": bot_response}

    return jsonify(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask ngrok API for Chat Bot')
    parser.add_argument(
        '--gpu-id',
        default=0,
        type=int,
        help='the ID of the GPU to run on'
    )
    parser.add_argument(
        '--model',
        default=f"{INFERENCE_DIR}/../huggingface_models/Pythia-Chat-Base-7B",
        help='name/path of the model'
    )
    parser.add_argument(
        '-g',
        '--gpu-vram',
        action='store',
        help='max VRAM to allocate per GPU',
        nargs='+',
        required=False,
    )
    parser.add_argument(
        '-r',
        '--cpu-ram',
        default=None,
        type=int,
        help='max CPU RAM to allocate',
        required=False
    )
    args = parser.parse_args()

    # Update parameters based on command line arguments
    model_name = args.model
    gpu_id = args.gpu_id

    # Set max_memory dictionary if given
    if args.gpu_vram is not None:
        max_memory = {}
        for vram in args.gpu_vram:
            gpu, memory = vram.split(':')
            max_memory[int(gpu)] = f"{memory}GiB"

    # Initialize the global chat_model instance
    chat_model = ChatModel(model_name, gpu_id, max_memory)

    # Run the Flask app
    app.run(port=12345)
