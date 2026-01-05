import json
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from transformers import AutoTokenizer
import os
import pickle
import copy
import argparse
from sortedcontainers import SortedDict

# prerequisites
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

class LRUCache:
    def __init__(self, max_token):
        self.cache = OrderedDict()
        self.MAX_TOKEN_SIZE = max_token
        self.current_token = 0

    def get(self, key):
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)  
            return self.cache[key]
        
    def put(self, key, token_count) -> None:
        evict_keys = []
        iter_cache_dict = iter(self.cache)

        while token_count + self.current_token > \
            self.MAX_TOKEN_SIZE:
            evict_key = next(iter_cache_dict)
            evict_token_size = self.cache[evict_key]
            self.current_token -= evict_token_size
            evict_keys.append(evict_key)

        for evict_key in evict_keys:
            del self.cache[evict_key]

        self.current_token += token_count
        self.cache[key] = token_count
        return evict_keys

class SortedDefaultDict(SortedDict):
    def __init__(self, default_factory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        self[key] = value = self.default_factory()
        return value

class Session:
    def __init__(self, session_id, current_time):
        self.session_id = session_id
        self.session_total_token = 0
        self.hit_token_count = 0
        self.access_history = []
        self.last_access_time = current_time
        self.chunk_dict = {}

    def check_chunk(self, chunk_id):
        if chunk_id not in self.chunk_dict:
            return -1
        else:
            hit_token_count += self.chunk_dict[chunk_id]
            return self.chunk_dict[chunk_id]

    def check_chunk_no_update(self, chunk_id):
        if chunk_id not in self.chunk_dict:
            return -1
        else:
            return self.chunk_dict[chunk_id]

    def add_chunk(self, chunk_id, token_count):
        self.chunk_dict[chunk_id] = token_count
        self.session_total_token += token_count

    def update_hit_token_count(self, hitted_token_count, current_time):
        self.hit_token_count += hitted_token_count
        self.last_access_time = current_time
        self.access_history.append((hitted_token_count, current_time))

    def calculate_score(self, current_time, session_visit_count, session_global_hit_count):
        score = 0
        decay_factor = 1
        for i, (hitted_token_count, timestamp) in enumerate(self.access_history):
            if i == len(self.access_history) - 1:
                time_diff = current_time - timestamp
                score += time_diff
            else:
                time_diff = self.access_history[i + 1][1] - timestamp
                score += time_diff

        score = time_diff * hitted_token_count/ (len(self.access_history) * (session_global_hit_count + 1) + len(self.chunk_dict))
        return score

class GreenCache:

    def __init__(self, max_token: int):
        self.score = SortedDefaultDict(OrderedDict)
        self.session_key_to_score = {}
        self.session_key_to_session = {}
        self.session_visit_count = {}
        self.session_global_hit_count = {}
        self.MAX_TOKEN_SIZE = max_token
        self.current_session_id = None
        self.global_timer = 0
        self.current_token = 0
        self.min_freq = 0

    def _get_session_id(self, key: str) -> str:
        return "_".join(key.split("_")[:2])

    def initialize_new_request(self, session_id: str, request_token_count: int):
        session_exist = -1
        self.current_session_id = session_id
        if session_id not in self.session_visit_count:
            self.session_visit_count[session_id] = 0 
        self.session_visit_count[session_id] += 1

        if self.current_session_id not in self.session_key_to_session:
            session = Session(self.current_session_id, self.global_timer)
            self.session_key_to_session[self.current_session_id] = session         
        else:
            session_exist = 1
        return session_exist

    def initialize_new_session_score(self, session_id: str, token_count: int):
        if session_id not in self.session_global_hit_count:
            self.session_global_hit_count[session_id] = 0 
        self.session_key_to_score[session_id] = token_count
        self.score[token_count][session_id] = None
        self.session_key_to_session[session_id].update_hit_token_count(token_count, self.global_timer-1)

    def get_no_update(self, key: str) -> str:
        get_session_id = self._get_session_id(key)
        if get_session_id not in self.session_key_to_session:
            return -1
        
        session = self.session_key_to_session[get_session_id]
        token_count = session.check_chunk_no_update(key)
        return token_count

    def update_score_revisit_request(self, session_id: str, hitted_token_count: int) -> str:
        if session_id not in self.session_global_hit_count:
            self.session_global_hit_count[session_id] = 0 
        self.session_global_hit_count[session_id] += hitted_token_count
        # append the new hit_history to the session
        self.session_key_to_session[session_id].update_hit_token_count(hitted_token_count, self.global_timer-1)
        new_score = self.session_key_to_session[session_id].calculate_score(self.global_timer, 
                                                                            self.session_visit_count[session_id],
                                                                            self.session_global_hit_count[session_id])
        session_score = self.session_key_to_score[session_id]
        self.score[session_score].pop(session_id)
        if len(self.score[session_score]) == 0:
            self.score.pop(session_score)
        self.score[new_score][session_id] = None
        self.session_key_to_score[session_id] = new_score


    def update_whole_cache_score(self):
        if len(self.session_key_to_session) == 0:
            return
        for session_id in self.session_key_to_session:
            new_score = self.session_key_to_session[session_id].calculate_score(self.global_timer, 
                                                                                self.session_visit_count[session_id],
                                                                                self.session_global_hit_count[session_id])
            session_score = self.session_key_to_score[session_id]
            self.score[session_score].pop(session_id)
            if len(self.score[session_score]) == 0:
                self.score.pop(session_score)
            self.score[new_score][session_id] = None
            self.session_key_to_score[session_id] = new_score

    def put(self, key: str, token_count: int) -> None:
        if self.MAX_TOKEN_SIZE == 0: return

        while self.current_token + token_count > self.MAX_TOKEN_SIZE:
            k = next(iter(self.score.peekitem(-1)[1]))
            if k == self._get_session_id(key):
                k, _ = self.score.peekitem(-2)[1].popitem(last=False)
                if len(self.score.peekitem(-2)[1]) == 0:
                    self.score.popitem(-2)
            else:
                k, _ = self.score.peekitem(-1)[1].popitem(last=False)
                if len(self.score.peekitem(-1)[1]) == 0:
                    self.score.popitem(-1)
            self.session_key_to_score.pop(k)
            pop_total_token = self.session_key_to_session.pop(k).session_total_token
            self.current_token -= pop_total_token

        put_session_id = self._get_session_id(key)
        self.session_key_to_session[put_session_id].add_chunk(key, token_count)
        self.current_token += token_count


def cache_simulation(chat_histories, size: int, slices: int, request_num: int, begining_index: int, output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3-70B-Instruct")

    def get_token_count(history):
        if len(history[:-2]) == 0:
            return 0
        return len(tokenizer.apply_chat_template(history[:-2], tokenize = True, add_generation_prompt = False)) - 1
    
    INT_0T = 0
    INT_1T = 3276800
    INT_2T = 6553600
    INT_4T = 13107200
    INT_5T = 16384000
    INT_8T = 26214400
    INT_10T = 32768000
    INT_12T = 39321600
    INT_16T = 52428800
    cache = None

    print(slices)

    requests_per_slice = request_num // slices
    for k in range(slices):
        match size:
            case 0:
                cache = GreenCache(INT_0T)
            case 1:
                cache = GreenCache(INT_1T)
            case 2:
                cache = GreenCache(INT_2T)
            case 4:
                cache = GreenCache(INT_4T)
            case 5:
                cache = GreenCache(INT_5T)
            case 8:
                cache = GreenCache(INT_8T)
            case 10:
                cache = GreenCache(INT_10T)
            case 12:
                cache = GreenCache(INT_12T)
            case 16:
                cache = GreenCache(INT_16T)
            case _:
                raise ValueError("invalid size value")

        hit_times = 0
        total_entries = []
        hit_entries = []
        nohit_entries = []
        hit_total_token = 0
        cache_list = []

        begining_index = begining_index + k * requests_per_slice
        for i, chat in enumerate(chat_histories):
            cache.global_timer += 1

            if i > 0 and i % 500 == 0:
                cache.update_whole_cache_score()

            session_exist = cache.initialize_new_request(chat.id, chat.prompt_token_count)

            if i == begining_index + requests_per_slice:
                break

            check_256_chunk_num = chat.prompt_token_count // 256
            suffix_length = chat.prompt_token_count % 256

            if i >= begining_index:
                real_time_check_256_chunk_num = get_token_count(chat.history) // 256
                hit_chunk_num = 0
                for j in range(real_time_check_256_chunk_num):
                    suffix_256_id = chat.id + f"_{j}_256"
                    if cache.get_no_update(suffix_256_id) != -1:
                        hit_chunk_num += 1
                if hit_chunk_num == real_time_check_256_chunk_num and len(chat.history) > 2:
                    hit_times += 1
                    hit_entries.append({i: chat.prompt_token_count})
                    cache_list.append('T')
                    hit_total_token += 256 * real_time_check_256_chunk_num
                else:
                    nohit_entries.append(i)
                    cache_list.append('F')
                total_entries.append(i)

            suffix_ids = []
            hit_token_count = 0
            should_put_suffix_count = 0
            should_put_suffix_256_count = 0

            for j in range(check_256_chunk_num):
                suffix_256_id = chat.id + f"_{j}_256"
                if cache.get_no_update(suffix_256_id) == -1:
                    suffix_ids.append((suffix_256_id, 256))
                    should_put_suffix_256_count += 256
                else:
                    hit_token_count += 256
            suffix_id = chat.id + f"_{int(len(chat.history)/2)}_{suffix_length}"

            if session_exist == 1:
                cache.update_score_revisit_request(chat.id, hit_token_count)

            for id in suffix_ids:
                cache.put(id[0], id[1])

            decoding_256_chunk_num = (chat.generated_total_token_count - 1) // 256
            decoding_ids = []
            for j in range(decoding_256_chunk_num):
                decoding_id = chat.id + f"_{j}_256"
                if cache.get_no_update(decoding_id) == -1:
                    decoding_ids.append((decoding_id, 256))
            
            if session_exist == -1:
                cache.initialize_new_session_score(chat.id, should_put_suffix_count + should_put_suffix_256_count + len(decoding_ids) * 256)

            for id in decoding_ids:
                cache.put(id[0], id[1])

        print(f"hit rate: {hit_times / requests_per_slice}")
        print(f"hit total token: {hit_total_token}")
        print(f"total entries: {len(total_entries)}")
        print(f"hit entries: {len(hit_entries)}")
        print(f"nohit entries: {len(nohit_entries)}")
        print(k)

        with open(output_dir + f"cache_list_{k}.txt", 'w') as f:
            f.write(','.join(cache_list))

        realTime_chat_histories = []
        for i in range(begining_index, begining_index + requests_per_slice):
            realTime_chat_histories.append(chat_histories[i])
        with open(output_dir + f'realTime_chat_histories_{k}.pickle', 'wb') as file:
            pickle.dump(realTime_chat_histories, file)

        cache_chat_histories = []
        for id in hit_entries:
            chat = copy.deepcopy(chat_histories[list(id.keys())[0]])
            chat.history = chat.history[:-2]
            chat.estimate_num_tokens()
            cache_chat_histories.append(chat)
        
        with open(output_dir + f'cache_chat_histories_{k}.pickle', 'wb') as file:
            pickle.dump(cache_chat_histories, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat-history', type=str, default='chat_history.pickle', help='chat history file')
    parser.add_argument('--size', type=int, default=16, help='cache size')
    parser.add_argument('--slices', type=int, default=1, help='cache slices')
    parser.add_argument('--request-num', type=int, default=500, help='request number')
    parser.add_argument('--begining-index', type=int, default=200000, help='begining index')
    parser.add_argument('--output-dir', type=str, default='./', help='output directory')
    args = parser.parse_args()

    with open(args.chat_history, 'rb') as file:
        chat_histories = pickle.load(file)
    print(len(chat_histories))

    cache_simulation(chat_histories, args.size, args.slices, args.request_num, args.begining_index, args.output_dir)

if __name__ == '__main__':
    main()