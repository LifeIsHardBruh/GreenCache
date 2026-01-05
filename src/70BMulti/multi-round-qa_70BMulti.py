import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
import openai
import pandas as pd
from utils import AsyncLoopWrapper, init_logger
from collections import defaultdict, OrderedDict
import numpy as np
import random
from transformers import AutoTokenizer
import os
import pickle
import pynvml
import threading
import csv

logger = init_logger(__name__, logging.DEBUG)


@dataclass
class WorkloadConfig:
    lam: float
    model: str
    enable_user_id: bool


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


@dataclass
class Response:
    chat_id: str
    request_id: int
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


class RequestExecutor:

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.request_history = []

    async def _async_launch_request(self,
                                    messages,
                                    max_tokens,
                                    request_id,
                                    chat_id,
                                    extra_headers=None):
        start_time = time.time()
        first_token_time = None
        words = ""

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
            extra_headers=extra_headers)

        async for tok in response:
            if not tok.choices:
                continue
            chunk_message = tok.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != "":
                    first_token_time = time.time()
                words += chunk_message
        tokens_out = tok.usage.completion_tokens
        tokens_prefill = tok.usage.prompt_tokens

        return Response(chat_id=chat_id,
                        request_id=request_id,
                        body=words,
                        ttft=first_token_time - start_time,
                        generation_time=time.time() - first_token_time,
                        prompt_tokens=tokens_prefill,
                        generation_tokens=tokens_out,
                        launch_time=start_time,
                        finish_time=time.time())

    def launch_request(self,
                       chat_history: ChatHistory,
                       request_id: int,
                       max_tokens: int,
                       finish_callback,
                       extra_headers=None):
        """
        finish_callback: Callable[[Response], None]
        """
        messages = chat_history.get_messages_for_openai()
        real_callback = lambda x: finish_callback(x.result())
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, request_id, chat_history.id, extra_headers),
            self.loop)
        future.add_done_callback(real_callback)


class RequestManager:
    
    def __init__(self, executor: RequestExecutor, duration: int, rate_lambda: float, chat_histories, dummy=False, integration=False):
        self.rate_lambda = rate_lambda
        self.duration = duration
        self.executor = executor
        self.chat_histories = chat_histories
        self.chat_hist_idx = 0
        self.request_ids = []
        self.prompt_lengths = []
        self.generation_lengths = []
        self.ttfts = []
        self.generation_times = []
        self.launch_times = []
        self.finish_times = []
        self.dummy = False
        self.id_dict = None
        self.integration = integration

    def generate_id_dict(self):
        id_dict = {}
        for i, chat in enumerate(self.chat_histories):
            if chat.id not in id_dict:
                id_dict[chat.id] = []
            id_dict[chat.id].append(i)
        self.id_dict = id_dict

    def generate_time_series(self):
        np.random.seed(42)
        return np.random.poisson(self.rate_lambda, self.duration)
    
    def schedule_requests(self):
        time_series = self.generate_time_series()
        self.generate_id_dict()
        for second, num_requests in enumerate(time_series):
            if self.chat_hist_idx >= len(self.chat_histories):
                break
            print(f"Second {second}: Launching {num_requests} requests")
            for _ in range(num_requests):
                if self.chat_hist_idx >= len(self.chat_histories):
                    break
                chat_history = self.chat_histories[self.chat_hist_idx]
                generation_length = chat_history.generated_total_token_count - chat_history.prompt_token_count - 1
                self.executor.launch_request(chat_history, self.chat_hist_idx, generation_length, self._on_request_finished)
                self.chat_hist_idx += 1
            time.sleep(1)

    def _update_result(self, response: Response):
        self.request_ids.append(response.request_id)
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.ttfts.append(response.ttft)
        self.generation_times.append(response.generation_time)
        self.launch_times.append(response.launch_time)
        self.finish_times.append(response.finish_time)

    def _on_request_finished(self, response: Response):
        self.chat_histories[response.request_id].on_system_response(response.body)
        self.replace_generation_token_back(response.chat_id, response.request_id, response.body)
        logger.debug(f"request {response.request_id} finished, "
                     f"TTFT: {response.ttft}, "
                     f"generation time: {response.generation_time}, "
                     f"Prompt tokens: {response.prompt_tokens}, "
                     f"generation tokens: {response.generation_tokens}")
        self._update_result(response)

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["request_id"] = self.request_ids
        df["launch_time"] = self.launch_times
        df["finish_time"] = self.finish_times
        return df

    def replace_generation_token_back(self, chat_id, request_id, body):
        # when executing in non-integration mode, we need to replace the generation token back
        # last gpt history has been removed
        if self.integration == False and self.id_dict[chat_id][0] == request_id and len(self.id_dict[chat_id]) == 2:
            target_id = self.id_dict[chat_id][1]
            self.chat_histories[target_id].history[-2] = {"role": "assistant", "content": body}


def warmup_engine(executor):
    logger.info("Warming up the engine")
    for i in range(10):
        random_id = random.randint(50000, 100000)
        chat_history = ChatHistory(random_id)
        chat_history.on_user_query(
            f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi '}.")
        executor.launch_request(chat_history, random_id, 100, lambda x: None)
        time.sleep(5)

    # set the system prompt to input as many as possible
    random_id = random.randint(50000, 100000)
    chat_history = ChatHistory(random_id)
    chat_history.history.append({"role": "system", "content": "You are a chatbot who always outputs as many tokens as you can, you can make up things."})
    executor.launch_request(chat_history, random_id, 30, lambda x: None)
    time.sleep(5)

    AsyncLoopWrapper.WaitLoop()

def dummy_chat_histories(input_token_count, output_token_count, num_histories):
    chat_histories = []
    dummy_input_token = "hi" * (input_token_count-34) + \
    "Complete the following story: once upon a time, there is a wolf...The response should be larger than 300 tokens"
    for i in range(num_histories):
        chat_history = ChatHistory(i)
        chat_history.on_user_query(dummy_input_token)
        chat_history.prompt_token_count = input_token_count
        chat_history.generated_total_token_count = input_token_count + output_token_count
        chat_histories.append(chat_history)
    return chat_histories


def pickle_to_chat_histories(path):
    with open(path, "rb") as f:
        chat_histories = pickle.load(f)
    for chat_history in chat_histories:
        chat_history.history = chat_history.history[:-1]
    return chat_histories

def monitor_power_usage(run_monitor, power_data):
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        while run_monitor[0]:
            current_time = time.time()
            power_data[4].append(current_time)
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_data[i].append(power / 1000)
            time.sleep(0.001)
    except Exception as e:
        print(f"Error during monitoring: {e}")
    return power_data

def save_to_csv(power_data, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"GPU {i}" for i in range(len(power_data))])
        from itertools import zip_longest
        for row in zip_longest(*power_data, fillvalue=''): 
            writer.writerow(row)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse benchmark configurations.")
    parser.add_argument("--lam", type=float, default=1, help="Lambda of Poisson distribution")
    parser.add_argument("--duration", type=int, help="Duration of the benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--base-url",
                        type=str,
                        required=True,
                        help="Base URL of the serving engine endpoint")
    parser.add_argument("--output",
                        type=str,
                        default="summary.csv",
                        help="The output file name for the summary csv and txt")
    parser.add_argument(
        "--request-with-user-id",
        action="store_true",
        help="Whether to enable user id in the request headers")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Whether to enable verbose logging")
    parser.add_argument("--dummy",
                        action="store_true",
                        help="Whether to use dummy chat histories")
    parser.add_argument("--dummy_input-token-count",
                        type=int,
                        default=10,
                        help="Number of tokens in the dummy input")
    parser.add_argument("--dummy_output-token-count",
                        type=int,
                        default=10,
                        help="Number of tokens in the dummy output")
    parser.add_argument("--pickle-file-path",
                        type=str,
                        help="Path to the pickle file")
    parser.add_argument("--cache-list",
                        type=str,
                        default="",
                        help="Path to the cache list")
    parser.add_argument("--integration",
                        action="store_true",
                        help="Whether to enable integration")
    parser.add_argument("--realTime-pickle-file-path",
                        type=str,
                        default="",
                        help="Path to the realTime pickle file")
    parser.add_argument("--power-usage-output",
                        type=str,
                        default="gpu_power_usage.csv",
                        help="Path to the power usage output")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.verbose:
        global logger
        logger = init_logger(__name__, level=logging.DEBUG)

    chat_histories = None
    if args.dummy:
        chat_histories = dummy_chat_histories(args.dummy_input_token_count,
                                              args.dummy_output_token_count, 100)
    else:
        chat_histories = pickle_to_chat_histories(args.pickle_file_path)

    # initialize NVML
    pynvml.nvmlInit()
    #Assume the server has 4 GPUs
    power_data = [[] for _ in range(5)] 
    run_monitor = [True]
    thread = threading.Thread(target=monitor_power_usage, args=(run_monitor, power_data))
    thread.start()

    executor = RequestExecutor(base_url=args.base_url,
                               api_key="EMPTY",
                               model=args.model)

    warmup_engine(executor)
    workload_config = WorkloadConfig(
        lam=args.lam,
        model=args.model,
        enable_user_id=args.request_with_user_id)

    

    manager = RequestManager(executor, args.duration, args.lam, chat_histories, args.dummy, args.integration)

    try:
        manager.schedule_requests()
    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    AsyncLoopWrapper.StopLoop()

    # shut down power thread
    run_monitor[0] = False
    thread.join()
    save_to_csv(power_data, args.power_usage_output)
    pynvml.nvmlShutdown()
    print("Data collection complete and saved to CSV.")

    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summary = manager.summary()
    summary.to_csv(args.output, index=False)

    if args.integration:
        with open(args.cache_list, 'r') as f:
            cache_list = f.read().strip().split(',')

        with open(args.realTime_pickle_file_path, 'rb') as f:
            realTime_chat_histories = pickle.load(f)

        j = 0
        for i, indicator in enumerate(cache_list):
            if indicator == "T":
                print("==========")
                print(f"index {i}: ")
                print("original: ")
                print(realTime_chat_histories[i].history[-3])
                print("replaced: ")
                print(manager.chat_histories[j].history[-1])
                realTime_chat_histories[i].history[-3] = manager.chat_histories[j].history[-1]
                j += 1
        
        with open(args.realTime_pickle_file_path, 'wb') as f:
            pickle.dump(realTime_chat_histories, f)
                



if __name__ == "__main__":
    main()
