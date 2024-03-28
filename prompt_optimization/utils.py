"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import config
import string


'''def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
    }
    retries = 0
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/chat/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=timeout
            )
            if r.status_code != 200:
                retries += 1
                time.sleep(1)
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1
    r = r.json()
    return [choice['message']['content'] for choice in r['choices']]


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "text-davinci-003",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=10
            )  
            if r.status_code != 200:
                time.sleep(2)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
    r = r.json()
    return r['choices']'''


import time
import requests
import config
import string
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_sectioned_prompt(s):
    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# chat_mistral 
#model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
def chatgpt(prompt, model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2", temperature=0.7, n=1, top_p=1, max_tokens=1024, stop=None,
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate responses
    outputs = model.generate(input_ids, 
                             max_length=max_tokens, 
                             temperature=temperature, 
                             top_p=top_p, 
                             num_return_sequences=n,
                             bad_words_ids=[[tokenizer.pad_token_id]],  # Stop if model generates padding token
                             no_repeat_ngram_size=3,  # Ensure no repeating n-grams
                             eos_token_id=tokenizer.eos_token_id,  # Stop if model generates EOS token
                             **logit_bias)

    # Decode generated responses
    decoded_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return decoded_responses

# model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

'''def chat_mistral(prompt, model_name_or_path, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# get_log_probs

def instructGPT_logprobs(prompt, model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2", temperature=0.7):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate log probabilities for the next token
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True, output_hidden_states=False, temperature=temperature, return_logits=True)

    # Extract logits for the next token
    next_token_logits = outputs.logits[:, -1, :]

    # Calculate log probabilities using softmax
    next_token_probs = torch.softmax(next_token_logits, dim=-1)

    # Get log probabilities
    next_token_log_probs = torch.log(next_token_probs)

    return next_token_log_probs







