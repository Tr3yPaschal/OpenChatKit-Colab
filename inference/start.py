import subprocess
import time

# Start the bot.py process with additional arguments
process = subprocess.Popen(
    ['python', 'inference/bot.py', 
     '--model', 'togethercomputer/RedPajama-INCITE-Base-3B-v1', 
     '-r', '16'], 
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE, 
    text=True, 
    bufsize=1,  # Line buffered
    universal_newlines=True)

prompt_encountered = False
while not prompt_encountered:
    # Read lines from the process output
    output_line = process.stdout.readline()
    print(output_line, end='')  # Print the output line
    if ">>>" in output_line:
        prompt_encountered = True

# Send the api_inference command to bot.py
command = 'api_inference {"prompt": "Hello, how are you?", "max_new_tokens": 50, "do_sample": true, "temperature": 0.7, "top_k": 40}\n'
process.stdin.write(command)
process.stdin.flush()

# Read the response after the command
response = process.stdout.readline()
print(response)

# Close the process
process.terminate()
