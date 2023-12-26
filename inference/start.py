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
    text=True)

# Give it some time to start up (2 minutes)
time.sleep(120)

# Send the api_inference command to bot.py
command = 'api_inference {"prompt": "Hello, how are you?", "max_new_tokens": 50, "do_sample": true, "temperature": 0.7, "top_k": 40}\n'
process.stdin.write(command)
process.stdin.flush()

# Read the output (may need adjustments based on how bot.py outputs data)
output = process.stdout.readline()
print(output)

# Close the process
process.terminate()
