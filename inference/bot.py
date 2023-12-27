import os
import sys
import socket
import argparse
import torch
import conversation as convo
import retrieval.wikipedia as wp
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList
from accelerate import infer_auto_device_map, init_empty_weights
from pyngrok import ngrok, conf

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: PYTHONPATH hacks are never a good idea. Clean this up later
sys.path.append(os.path.join(INFERENCE_DIR, '..'))

# Define constants for flags
START_FLAG = "__START__"
READY_FLAG = "__READY__"
END_FLAG = "__END__"


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
        device = torch.device('cuda', gpu_id)   # TODO: allow sending to CPU

        # recommended default for devices with > 40 GB VRAM
        # load model onto one device
        if max_memory is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto")
            self._model.to(device)
        # load the model with the given max_memory config (for devices with insufficient VRAM or multi-gpu)
        else:
            config = AutoConfig.from_pretrained(model_name)
            # load empty weights
            with init_empty_weights():
                model_from_conf = AutoModelForCausalLM.from_config(config)

            model_from_conf.tie_weights()

            # create a device_map from max_memory
            device_map = infer_auto_device_map(
                model_from_conf,
                max_memory=max_memory,
                no_split_module_classes=["GPTNeoXLayer"],
                dtype="float16"
            )
            # load the model with the above device_map
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                offload_folder="offload",  # optional offload-to-disk overflow directory (auto-created)
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

        print(START_FLAG)
        output = output[len(prompt):]

        return output


class OpenChatKitShell:
    def __init__(self, gpu_id, model_name_or_path, max_tokens, sample, temperature, top_k, retrieval, max_memory, do_stream):
        self._gpu_id = gpu_id
        self._model_name_or_path = model_name_or_path
        self._max_tokens = max_tokens
        self._sample = sample
        self._temperature = temperature
        self._top_k = top_k
        self._retrieval = retrieval
        self._max_memory = max_memory
        self._do_stream = do_stream

        # Create a socket server
        self.HOST = "127.0.0.1"  # Set your ngrok tunnel URL (127.0.0.1 in this case)
        self.PORT = 12345  # Set your ngrok tunnel port (12345 in this case)
        self._server_socket = None
        self._client_socket = None

    def start_server(self):
        # Initialize the server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind((self.HOST, self.PORT))
        self._server_socket.listen()

        print(READY_FLAG)

        # Accept a connection from the client
        self._client_socket, _ = self._server_socket.accept()

        # Start processing messages from the client
        self.process_messages()

    def process_messages(self):
        # Initialize conversation components
        model = ChatModel(self._model_name_or_path, self._gpu_id, self._max_memory)
        convo = convo.Conversation(model.human_id, model.bot_id)

        while True:
            message = self._client_socket.recv(1024).decode()
            if not message:
                break

            # Check for the __END__ flag to exit
            if message == END_FLAG:
                break

            # Check for the __START__ flag to indicate the start of a new conversation
            if message == START_FLAG:
                convo = convo.Conversation(model.human_id, model.bot_id)
                continue

            # Process the message
            if self._retrieval:
                results = model._index.search(message)
                if len(results) > 0:
                    convo.push_context_turn(results[0])

            convo.push_human_turn(message)

            output = model.do_inference(
                convo.get_raw_prompt(),
                self._max_tokens,
                self._sample,
                self._temperature,
                self._top_k,
                lambda x: self._client_socket.send(x.encode()) if self._do_stream else None,
            )

            convo.push_model_response(output)

            response = "" if self._do_stream else convo.get_last_turn()

            # Send the response back to the client
            self._client_socket.send(response.encode())

    def close_connections(self):
        if self._client_socket:
            self._client_socket.close()
        if self._server_socket:
            self._server_socket.close()

    def run(self):
        try:
            # Start the server
            self.start_server()
        except Exception as e:
            print("An error occurred:", str(e))
        finally:
            self.close_connections()


def main():
    parser = argparse.ArgumentParser(
        description='Test harness for OpenChatKit')

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
        '--max-tokens',
        default=128,
        type=int,
        help='the maximum number of tokens to generate'
    )
    parser.add_argument(
        '--sample',
        default=True,
        action='store_true',
        help='indicates whether to sample'
    )
    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='indicates whether to stream tokens'
    )
    parser.add_argument(
        '--temperature',
        default=0.6,
        type=float,
        help='temperature for the LM'
    )
    parser.add_argument(
        '--top-k',
        default=40,
        type=int,
        help='top-k for the LM'
    )
    parser.add_argument(
        '--retrieval',
        default=False,
        action='store_true',
        help='augment queries with context from the retrieval index'
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
    parser.add_argument(
        '--ngrok-host',
        default='localhost',
        help='host for ngrok tunnel'
    )
    parser.add_argument(
        '--ngrok-port',
        default=5000,
        type=int,
        help='port for ngrok tunnel'
    )
    args = parser.parse_args()

    # Set max_memory dictionary if given
    if args.gpu_vram is None:
        max_memory = None
    else:
        max_memory = {}
        for i in range(len(args.gpu_vram)):
            # Assign CUDA ID as label and XGiB as value
            max_memory[int(args.gpu_vram[i].split(':')[0])] = f"{args.gpu_vram[i].split(':')[1]}GiB"

        if args.cpu_ram is not None:
            # Add CPU to max-memory if given
            max_memory['cpu'] = f"{int(args.cpu_ram)}GiB"

    # Initialize and run the chat server
    chat_server = OpenChatKitShell(
        args.gpu_id,
        args.model,
        args.max_tokens,
        args.sample,
        args.temperature,
        args.top_k,
        args.retrieval,
        max_memory,
        not args.no_stream,
    )
    chat_server.run()


if __name__ == '__main__':
    main()