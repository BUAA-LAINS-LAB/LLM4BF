import json
from datasets import load_dataset
import ast
from isac_utils import *
import re
import random


SYSTEM_PROMPT = (
    "You are a large language model specialized in solving downlink ISAC beamforming "
    "Below is an instruction describing this optimization problem. "
)

RESPONSE_TEMPLATE = "\n# Response:\n"

SUCCESS = 0
FORMAT_ERROR = 1
FEASIBILITY_ERROR = 2


FEAS_WEIGHTS = {
    "parse": 0.5,
    "shape": 0.3,
    "sinr": 0.1,
    "power": 0.1,
}


def round_floats(obj, ndigits=3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [round_floats(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def get_dataset(data_dir):
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        user_nums = examples["num_users"]


        prompts = []
        for instruction, input_text, user_num in zip(instructions, inputs, user_nums):
            input_rounded = round_floats(input_text, ndigits=3)
            input_str = json.dumps(input_rounded, ensure_ascii=False, separators=(",", ":"))


            text = (
                    SYSTEM_PROMPT
                    + "\n# Users:\n"
                    + str(user_num)
                    + "\n# Instruction:\n"
                    + instruction
                    + "\n# Input:\n"
                    + input_str
                    + RESPONSE_TEMPLATE
            )
            prompts.append(text)

        return {"prompt": prompts}

    raw_datasets = load_dataset("json", data_files=data_dir, split="train")


    map_dataset = raw_datasets.map(
        formatting_prompts_func,
        batched=True,
        load_from_cache_file=False,
        remove_columns=[]
    )

    return map_dataset


def parse_w_from_pred(gen_text, K, config):
    match = re.search(r"\[([^\]]+)\]$", gen_text)
    if not match:
        return FORMAT_ERROR, None


    inner = match.group(1)


    raw_tokens = inner.split(",")
    values = []
    for token in raw_tokens:
        token = token.strip()
        if token == "":
            continue
        try:
            values.append(float(token))
        except ValueError:
            return FORMAT_ERROR, None


    Nt = config.Nt
    expect_length = K * 2 * Nt
    parsed_len = len(values)
    length_fixed = (parsed_len != expect_length)

    if parsed_len >= expect_length:
        llm_output = values[:expect_length]
    else:
        llm_output = values + [0.0] * (expect_length - parsed_len)

    status = SUCCESS if not length_fixed else FEASIBILITY_ERROR
    return status, llm_output


def feasibility_power(W_stack, PT, K, power_tolerance=1e-1):
    total_power = 0.0
    for k in range(K):
        total_power += float(np.real(np.trace(W_stack[k])))

    if total_power <= PT + power_tolerance:
        power_ratio = 1.0
    else:
        power_ratio = max(0.0, min(1.0, PT / (total_power + 1e-12)))
    return power_ratio


def feasibility_SINR(K, config, H, Gamma_dB, W_stack, sinr_tolerance=1e-1):
    Gamma = config.Gamma(Gamma_dB)
    Nt = config.Nt

    Q = np.zeros((K, Nt, Nt), dtype=np.complex128)
    for k in range(K):
        hk = H[k].reshape(Nt, 1)
        Q[k] = hk @ hk.conj().T

    sinr_satisfied = 0
    for k in range(K):
        Qk = Q[k]


        signal_k = float(np.real(np.trace(Qk @ W_stack[k])))


        interf_k = 0.0
        for j in range(K):
            if j == k:
                continue
            interf_k += float(np.real(np.trace(Qk @ W_stack[j])))

        lhs = signal_k - Gamma * interf_k
        rhs = Gamma * config.sigma2C

        if lhs + sinr_tolerance >= rhs:
            sinr_satisfied += 1

    if K > 0:
        sinr_ratio = sinr_satisfied / K
    else:
        sinr_ratio = 0.0
    return sinr_ratio


def selected_eval_dataset(num_samples, eval_dataset):
    n = min(num_samples, len(eval_dataset))
    indices = random.sample(range(len(eval_dataset)), n)
    return eval_dataset.select(indices)
