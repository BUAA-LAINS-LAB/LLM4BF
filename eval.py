from unsloth import FastLanguageModel
import argparse
from utils import *
import random
import torch
from peft import PeftModel
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Tester for our optimization LLM')


    parser.add_argument('--model_path', type=str,
                        default='../LLM4BF-release/rl/k3',
                        help='Model path')
    parser.add_argument('--max_completion_length', type=int, default=1000, help='Maximum completion length')
    parser.add_argument('--model_method', type=str, default='lora', choices=['lora', 'full'],
                        help='use lora or full LLM for evaluation')
    parser.add_argument('--base_model', type=str,
                        default='./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit', help='Model name')
    parser.add_argument('--seed', type=int, default=3047, help='Random seed')


    parser.add_argument('--eval_method', type=str, default='best_of_n_conditions',
                        choices=['vanilla_fast', 'vanilla_conditions', 'best_of_n_fast', 'best_of_n_conditions'],
                        help='Evaluation method')


    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to evaluate (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='Use 4-bit quantization to reduce memory usage')


    parser.add_argument('--best_of_n', type=int, default=4,
                        help='Number of solutions to generate per prompt (best_of_n only)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling parameter (best_of_n only)')


    parser.add_argument('--dataset_eval_path', type=str, default='dataset/ISAC_Dataset_Wk_3_eval.json',
                        help='the path for evaluation dataset')

    args = parser.parse_args()

    return args


config = ISACConfig()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(args):
    if args.model_method == "lora":
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            local_files_only=True,
            dtype=args.dtype,
        )


        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            local_files_only=True,
            dtype=args.dtype,
        )


    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def evaluate_vanilla_fast(args, model, tokenizer, eval_dataset, batch_size):
    subset = selected_eval_dataset(args.num_samples, eval_dataset)
    mse_list = []
    valid_list = []

    model.eval()

    for start in tqdm(range(0, len(subset), batch_size), desc="Evaluating vanilla (batched)"):
        end = min(start + batch_size, len(subset))
        batch = subset[start:end]

        prompts = batch["prompt"]
        inputs_obj = batch["input"]
        Ks = batch["num_users"]
        objs = batch["objective"]


        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)


        with torch.no_grad():
            gen_out = model.generate(
                **inputs, temperature=args.temperature,
                max_new_tokens=args.max_completion_length, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)


        gen_tokens = gen_out[:, inputs["input_ids"].shape[1]:]
        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)


        for gen_text, input_obj, K, obj in zip(gen_texts, inputs_obj, Ks, objs):
            K = int(K)
            status, llm_output = parse_w_from_pred(gen_text, K, config)

            if status == FORMAT_ERROR or llm_output is None:
                valid_list.append(0)
                continue

            W_stack = vectors_to_W_stack(llm_output, config, K)
            valid_list.append(1)

            theta = float(input_obj["theta"])
            CRB_pred = compute_crb_for_sample(config, theta, W_stack)
            CRB_gt = float(obj)

            diff = float(CRB_pred) - float(CRB_gt)
            mse = diff * diff
            mse_list.append(round(mse, 4))

    return mse_list, valid_list


def evaluate_vanilla_all_conditions(args, model, tokenizer, eval_dataset, batch_size):
    subset = selected_eval_dataset(args.num_samples, eval_dataset)
    mse_list = []
    valid_list = []

    model.eval()

    for start in tqdm(range(0, len(subset), batch_size), desc="Evaluating vanilla (batched)"):
        end = min(start + batch_size, len(subset))
        batch = subset[start:end]

        prompts = batch["prompt"]
        inputs_obj = batch["input"]
        Ks = batch["num_users"]
        objs = batch["objective"]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            gen_out = model.generate(
                **inputs, temperature=args.temperature,
                max_new_tokens=args.max_completion_length, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

        gen_tokens = gen_out[:, inputs["input_ids"].shape[1]:]
        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for gen_text, input_obj, K, obj in zip(gen_texts, inputs_obj, Ks, objs):
            valid_score = 0

            K = int(K)
            status, llm_output = parse_w_from_pred(gen_text, K, config)

            if status == FORMAT_ERROR or llm_output is None:
                valid_score = 0.0
                valid_list.append(valid_score)
                continue

            valid_score += FEAS_WEIGHTS["parse"]


            if status == FEASIBILITY_ERROR:
                valid_list.append(valid_score)
                continue

            valid_score += FEAS_WEIGHTS["shape"]

            W_stack = vectors_to_W_stack(llm_output, config, K)


            power_ratio = feasibility_power(W_stack, input_obj["PT"], K)
            valid_score += FEAS_WEIGHTS["power"] * power_ratio


            H = compute_channel_H(input_obj)
            sinr_ratio = feasibility_SINR(K, config, H, input_obj["Gamma"], W_stack)
            valid_score += FEAS_WEIGHTS["sinr"] * sinr_ratio

            valid_list.append(valid_score)

            theta = float(input_obj["theta"])
            CRB_pred = compute_crb_for_sample(config, theta, W_stack)
            CRB_gt = float(obj)

            diff = float(CRB_pred) - float(CRB_gt)
            mse = diff * diff
            mse_list.append(round(mse, 4))

    return mse_list, valid_list


def evaluate_best_of_n_fast(args, model, tokenizer, eval_dataset, batch_size):
    model.eval()
    subset = selected_eval_dataset(args.num_samples, eval_dataset)

    mse_list = []
    valid_list = []
    n_candidates = args.best_of_n

    for start in tqdm(range(0, len(subset), batch_size), desc="Evaluating best-of-n (batched)"):
        end = min(start + batch_size, len(subset))
        batch = subset[start:end]

        prompts = batch["prompt"]
        inputs_obj = batch["input"]
        Ks = batch["num_users"]
        objs = batch["objective"]

        current_bs = len(prompts)

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)


        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=args.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=n_candidates,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )


        prompt_len = enc["input_ids"].shape[1]
        gen_tokens = gen_out[:, prompt_len:]
        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)


        for i in range(current_bs):
            input_obj = inputs_obj[i]
            K = int(Ks[i])
            CRB_gt = float(objs[i])


            cand_texts = gen_texts[i * n_candidates: (i + 1) * n_candidates]

            best_CRB = None
            best_mse = None

            any_valid = False

            for gen_text in cand_texts:
                status, llm_output = parse_w_from_pred(gen_text, K, config)
                if status == FORMAT_ERROR or llm_output is None:
                    continue


                any_valid = True
                W_stack = vectors_to_W_stack(llm_output, config, K)

                theta = float(input_obj["theta"])
                CRB_pred = compute_crb_for_sample(config, theta, W_stack)
                if (best_CRB is None) or (CRB_pred < best_CRB):
                    best_CRB = CRB_pred
                    diff = float(CRB_pred) - CRB_gt
                    best_mse = diff * diff

            if not any_valid:
                valid_list.append(0)
            else:
                valid_list.append(1)
                mse_list.append(round(best_mse, 4))

    return mse_list, valid_list


def evaluate_best_of_n_all_conditions(args, model, tokenizer, eval_dataset, batch_size):
    model.eval()

    subset = selected_eval_dataset(args.num_samples, eval_dataset)

    mse_list = []
    valid_list = []
    n_candidates = args.best_of_n

    for start in tqdm(range(0, len(subset), batch_size), desc="Evaluating best-of-n (all conds, batched)"):
        end = min(start + batch_size, len(subset))
        batch = subset[start:end]

        prompts = batch["prompt"]
        inputs_obj = batch["input"]
        Ks = batch["num_users"]
        objs = batch["objective"]

        current_bs = len(prompts)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=args.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=n_candidates,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )


        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = gen_out[:, prompt_len:]
        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)


        for i in range(current_bs):
            input_obj = inputs_obj[i]
            K = int(Ks[i])
            CRB_gt = float(objs[i])

            cand_texts = gen_texts[i * n_candidates: (i + 1) * n_candidates]

            best_CRB = None
            best_mse = None
            valid_of_best_CRB = None
            best_valid_any = 0.0

            for gen_text in cand_texts:
                status, llm_output = parse_w_from_pred(gen_text, K, config)

                if status == FORMAT_ERROR or llm_output is None:
                    cand_valid = 0.0
                    best_valid_any = max(best_valid_any, cand_valid)
                    continue

                cand_valid = 0.0
                cand_valid += FEAS_WEIGHTS["parse"]

                if status == FEASIBILITY_ERROR:
                    best_valid_any = max(best_valid_any, cand_valid)
                    continue

                cand_valid += FEAS_WEIGHTS["shape"]
                W_stack = vectors_to_W_stack(llm_output, config, K)

                power_ratio = feasibility_power(W_stack, input_obj["PT"], K)
                cand_valid += FEAS_WEIGHTS["power"] * power_ratio

                H = compute_channel_H(input_obj)
                sinr_ratio = feasibility_SINR(K, config, H, input_obj["Gamma"], W_stack)
                cand_valid += FEAS_WEIGHTS["sinr"] * sinr_ratio

                best_valid_any = max(best_valid_any, cand_valid)

                theta = float(input_obj["theta"])
                CRB_pred = compute_crb_for_sample(config, theta, W_stack)
                diff = float(CRB_pred) - CRB_gt
                cand_mse = diff * diff

                if (best_CRB is None) or (CRB_pred < best_CRB):
                    best_CRB = CRB_pred
                    best_mse = cand_mse
                    valid_of_best_CRB = cand_valid

            if best_CRB is not None:
                valid_list.append(valid_of_best_CRB)
                mse_list.append(round(best_mse, 4))
            else:
                valid_list.append(best_valid_any)

    return mse_list, valid_list


def evaluate_model(args):
    model, tokenizer = load_model_and_tokenizer(args)

    eval_dataset = get_dataset(args.dataset_eval_path)

    if args.eval_method == 'vanilla_fast':
        return evaluate_vanilla_fast(args, model, tokenizer, eval_dataset, args.batch_size)
    elif args.eval_method == 'vanilla_conditions':
        return evaluate_vanilla_all_conditions(args, model, tokenizer, eval_dataset, args.batch_size)
    elif args.eval_method == 'best_of_n_fast':
        return evaluate_best_of_n_fast(args, model, tokenizer, eval_dataset, args.batch_size)
    else:
        return evaluate_best_of_n_all_conditions(args, model, tokenizer, eval_dataset, args.batch_size)


def main(args):
    set_global_seed(args.seed)

    print(f"Running {args.eval_method} evaluation for {args.model_path}")
    print(f"Number of samples: {args.num_samples}")

    if args.eval_method == 'best_of_n':
        print(f"Best-of-N: {args.best_of_n}")
        print(f"Batch size: {args.batch_size}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")

    mse_list, valid_list = evaluate_model(args)


    if mse_list:
        overall_mse = float(np.mean(mse_list))
    else:
        overall_mse = float("nan")

    overall_valid = float(np.mean(valid_list))
    print(f"\nOverall valid percent for {args.num_samples}")
    print(f"\nvalid_ratio = {overall_valid: .3f}, MSE = {overall_mse: .3f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
