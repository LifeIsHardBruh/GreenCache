import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="Process data percentage.")
parser.add_argument(
    "--parse",
    type=float,
    default=1,
    help="The percentage of data to process (0 to 1). Default is 1 (100%).")

args = parser.parse_args()

with open('ShareGPT_V3_unfiltered_cleaned_split.json', 'r',
          encoding='utf-8') as file:
    data = json.load(file)


def estimate_num_tokens(text: dict, isUser: bool) -> int:
    if not hasattr(estimate_num_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        estimate_num_tokens.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct")
    if isUser:
        return len(estimate_num_tokens.tokenizer.apply_chat_template(text, tokenize = True, add_generation_prompt = True))
    else:
        return len(estimate_num_tokens.tokenizer.apply_chat_template(text, tokenize = True, add_generation_prompt = False))

def has_consecutive_humans(conversation):
    for i in range(len(conversation) - 1):
        if 'human' in conversation[i]['from'] and 'human' in conversation[i + 1]['from']:
            return True
    return False

def has_consecutive_gpt(conversation):
    for i in range(len(conversation) - 1):
        if 'gpt' in conversation[i]['from'] and 'gpt' in conversation[i + 1]['from']:
            return True
    return False

num_of_ids = len(data)
print(f"Number of IDs: {num_of_ids}")
data = data[:int(num_of_ids * args.parse)]
data_output = []
count = 0

for d in data:
    if has_consecutive_humans(d['conversations']) or has_consecutive_gpt(d['conversations']):
        count += 1
        print(f"Consecutive humans/gpt in {count}")
        continue

    d_output = {}
    d_output['id'] = d['id']
    d_output['conversations'] = []

    try:
        text = []
        human_first = False
        conversation_len = len(d['conversations'])
        for i, conv in enumerate(d['conversations']):
            if conv['from'] == 'human':
                human_first = True
            elif conv['from'] == 'gpt' and not human_first:
                continue

            if conv['from'] == 'human' and i == conversation_len - 1:
                continue

            if conv['from'] == 'human':
                text.append({"role": "user", "content": conv['value']})
                token_number = estimate_num_tokens(text, True)
                if len(conv['value']) == 0:
                    raise ValueError("Token number is 0")
                d_output['conversations'].append(conv)
                conv['num_tokens'] = token_number
            elif conv['from'] == 'gpt':
                text.append({"role": "assistant", "content": conv['value']})
                token_number = estimate_num_tokens(text, False)
                if len(conv['value']) == 0:
                    raise ValueError("Token number is 0")
                d_output['conversations'].append(conv)
                conv['num_tokens'] = token_number
            else:
                raise ValueError("Unknown speaker")
    except:
        count += 1
        print(f"Unknown speaker/Token number is 0 in {count}")
        continue
    

    d_output['num_round'] = len(
        d_output['conversations'])  # human is one round, gpt is another round

    count += 1
    print(f"Finished {count}")

    #skip the session with no conversation
    if len(d_output['conversations']) == 0:
        continue

    data_output.append(d_output)

with open('ShareGPT_8B_real_cleaned_with_token_length.json', 'w', encoding='utf-8') as file:
    json.dump(data_output, file, ensure_ascii=False, indent=2)