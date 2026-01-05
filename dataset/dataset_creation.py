import json
from collections import defaultdict, OrderedDict, deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from transformers import AutoTokenizer
import os
import pickle
import copy

class ChatHistory:
    tokenizer = None

    def __init__(self, id):
        self.id = id
        self.history = []
        self.prompt_token_count = 0
        self.generated_total_token_count = 0

    def on_user_query(self, query: str):
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": query})
        else:
            assert self.history[-1][
                "role"] == "assistant", "Expect system response"
            self.history.append({"role": "user", "content": query})

    def on_system_response(self, response: str):
        assert len(self.history) > 0, "Expect user query"
        assert self.history[-1]["role"] == "user", "Expect user query"
        self.history.append({"role": "assistant", "content": response})

    def get_messages_for_openai(self):
        return self.history

    def estimate_num_tokens(self):
        if ChatHistory.tokenizer is None:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            ChatHistory.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-70B-Instruct")

        self.prompt_token_count = \
        len(self.tokenizer.apply_chat_template(self.history[:-1], tokenize = True, add_generation_prompt = True))

        self.generated_total_token_count = \
        len(self.tokenizer.apply_chat_template(self.history, tokenize = True, add_generation_prompt = False))

    def __len__(self):
        return len(self.history)
    
with open('ShareGPT_8B_real_cleaned_with_token_length.json', 'r') as f:
    dataset = json.load(f)

dataset_dict = {d['id']: d for d in dataset}

prefix_lengths = []
for i, conv in enumerate(dataset):
    prefix_len = 0
    num_turn = len(conv['conversations'])
    for j, turn in enumerate(conv['conversations']):
        try:
            prefix_len += turn['num_tokens']
            if j == num_turn - 1:
                prefix_lengths.append(prefix_len)
        except KeyError:
            print(turn)
            continue

sampled_data_dict = {d['id']: d for d in dataset}

all_turn_count = 0
session_dict = {}
session_list = []
for d in dataset:
    all_turn_count += len(d['conversations'])
    session_dict[d['id']] = len(d['conversations'])
    session_list.append(d['id'])

print(all_turn_count)

#create the trace sequence
sequence = []
random.seed(42)
while len(session_dict) != 0:
    picked = random.choice(session_list)
    sequence.append(picked)
    session_dict[picked] -= 2
    if session_dict[picked] == 0:
        del session_dict[picked]
        session_list.remove(picked)

seq_arr = np.array(sequence)
random.seed(42)
random.shuffle(seq_arr)
shuffled_sequence = seq_arr.tolist()

with open('list.txt', 'w', encoding='utf-8') as file:
    for item in shuffled_sequence:
        file.write(item + '\n')

chat_histories_dict = {}
chat_histories = []
for k, seq_id in enumerate(shuffled_sequence):
    if seq_id not in chat_histories_dict:
        chat_histories_dict[seq_id] = 0
        current_chat_history = ChatHistory(seq_id)
        current_chat_history.on_user_query(sampled_data_dict[seq_id]['conversations'][0]['value'])
        current_chat_history.on_system_response(sampled_data_dict[seq_id]['conversations'][1]['value'])
        current_chat_history.estimate_num_tokens()
        chat_histories.append(current_chat_history)
        chat_histories_dict[seq_id] += 1
    else:
        current_chat_history = ChatHistory(seq_id)
        for i in range(chat_histories_dict[seq_id]):
            current_chat_history.on_user_query(sampled_data_dict[seq_id]['conversations'][2*i]['value'])
            current_chat_history.on_system_response(sampled_data_dict[seq_id]['conversations'][2*i + 1]['value'])
        current_chat_history.on_user_query(sampled_data_dict[seq_id]['conversations'][2*chat_histories_dict[seq_id]]['value'])
        current_chat_history.on_system_response(sampled_data_dict[seq_id]['conversations'][2*chat_histories_dict[seq_id] + 1]['value'])
        current_chat_history.estimate_num_tokens()
        chat_histories.append(current_chat_history)
        chat_histories_dict[seq_id] += 1
    print(f'{k} chat histories created')

with open('chat_history_full_dataset.pickle', 'wb') as file:
    pickle.dump(chat_histories, file)