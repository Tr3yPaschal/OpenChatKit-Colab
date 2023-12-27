import subprocess
import json
import time

process = subprocess.Popen(
    ['python', 'inference/bot.py',
     '--model', 'togethercomputer/RedPajama-INCITE-Base-3B-v1',
     '-r', '16'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    universal_newlines=True)

prompt_encountered = False
while not prompt_encountered:
    output_line = process.stdout.readline()
    print(output_line, end='')
    if "READY" in output_line:
        prompt_encountered = True

def send_api_request(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=40):
    command = 'api_inference ' + json.dumps({
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k
    }) + '\n'
    process.stdin.write(command)
    process.stdin.flush()
    response = process.stdout.readline()
    print(response)

# Example usage
send_api_request("Hello, how are you?")

# Keep the process open for further API requests
# Implement a loop or a mechanism to accept new API requests here
# ...

process.terminate()