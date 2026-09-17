import argparse
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling
import warnings
from datetime import datetime
import torch
from utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Trainer for solving ISAC optimization problem')


    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='Use 4-bit quantization to reduce memory usage')
    parser.add_argument('--local_file', action='store_true', default=True,
                        help='Load LLM pretrained checkpoint from local path')
    parser.add_argument('--model_name', type=str,
                        default='./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit', help='Model name')
    parser.add_argument('--data_dir', type=str,
                        default='dataset/ISAC_Dataset_Wk_1_SFT.json', help='Dataset path')


    parser.add_argument('--lora_r', type=int, default=16, help='Rank of the LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Scaling factor for LoRA updates')
    parser.add_argument('--bias', type=str, default='lora_only', choices=['none', 'all', 'lora_only'],
                        help='Bias type, if use eval_adapter, set this options **none**')
    parser.add_argument('--use_rslora', action='store_true', default=False, help='Use RSLoRA')
    parser.add_argument('--loftq_config', type=str, default=None, help='LoFT-Q configuration')


    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth', help='Use gradient checkpointing')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per device during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=20, help='Number of warmup steps')
    parser.add_argument('--num_train_epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--optim', type=str, default='adamw_8bit', help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=50, help='Total save limit for model checkpoints')
    parser.add_argument('--save_step', type=int, default=500, help='Steps interval to save model checkpoints')
    parser.add_argument('--train_lm_head', action='store_true', default=False,
                        help='Whether to train the language model head or not')
    parser.add_argument('--train_embed_tokens', action='store_true', default=False,
                        help='Whether to train the embed_tokens or not')
    parser.add_argument('--output_dir', type=str, default='./history', help='Output directory name')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume the model from checkpoint')


    args = parser.parse_args()

    return args


config = ISACConfig()


class SafeDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, mlm=False, fallback_strategy="last_portion"):
        super().__init__(tokenizer=tokenizer, mlm=mlm)


        self.response_template = response_template
        self.tokenizer = tokenizer


        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.fallback_strategy = fallback_strategy

    def torch_call(self, examples):
        batch = super().torch_call(examples)


        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i].tolist()


            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            char_pos = text.find(self.response_template)

            if char_pos != -1:
                prefix_text = text[: char_pos + len(self.response_template)]


                prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                response_start_idx = len(prefix_ids)


                batch["labels"][i, :response_start_idx] = -100
                continue

            warnings.warn(f"Response template not found in example {i}")

            if self.fallback_strategy == "last_portion":
                seq_length = len(input_ids)
                start_pos = int(0.9 * seq_length)
                batch["labels"][i, :start_pos] = -100

            elif self.fallback_strategy == "full_example":
                pass

            elif self.fallback_strategy == "skip":
                batch["labels"][i, :] = -100

        return batch


def get_isac_sft_datasets(tokenizer, data_files):
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        user_nums = examples["num_users"]


        texts = []
        for instruction, input_text, output_text, user_num in zip(
                instructions, inputs, outputs, user_nums
        ):
            input_rounded = round_floats(input_text, ndigits=3)
            output_rounded = round_floats(output_text, ndigits=3)


            input_str = json.dumps(input_rounded, ensure_ascii=False, separators=(",", ":"))
            output_str = json.dumps(output_rounded, ensure_ascii=False, separators=(",", ":"))


            text = (
                    SYSTEM_PROMPT
                    + "\n# Users:\n"
                    + str(user_num)
                    + "\n# Instruction:\n"
                    + instruction
                    + "\n# Input:\n"
                    + input_str
                    + RESPONSE_TEMPLATE
                    + output_str
                    + EOS_TOKEN
            )
            texts.append(text)

        return {"text": texts}

    raw_datasets = load_dataset("json", data_files=data_files, split="train")

    train_dataset = raw_datasets.map(
        formatting_prompts_func,
        batched=True,
        load_from_cache_file=False,
        remove_columns=[],
    )

    data_collator = SafeDataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
        fallback_strategy="last_portion",
    )

    return train_dataset, data_collator


def get_isac_sft_datasets_numeric_only(tokenizer, data_files):
    EOS_TOKEN = tokenizer.eos_token

    def flatten_2d(mat):
        return [x for row in mat for x in row]

    def fmt(v):
        return f"{float(v): .3f}"

    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        user_nums = examples["num_users"]

        texts = []
        for inp, out, K in zip(inputs, outputs, user_nums):
            theta = inp["theta"]
            PT = inp["PT"]
            Gamma = inp["Gamma"]
            H_real = inp["H_real"]
            H_imag = inp["H_imag"]


            H_real_flat = flatten_2d(H_real)
            H_imag_flat = flatten_2d(H_imag)

            x_vec = [K, theta, PT, Gamma, *H_real_flat, *H_imag_flat]
            y_vec = out


            input_str = " ".join(fmt(v) for v in x_vec)
            output_str = " ".join(fmt(v) for v in y_vec)


            text = input_str + RESPONSE_TEMPLATE + output_str + EOS_TOKEN
            texts.append(text)

        return {"text": texts}

    raw_datasets = load_dataset("json", data_files=data_files, split="train")

    train_dataset = raw_datasets.map(
        formatting_prompts_func,
        batched=True,
        load_from_cache_file=False,
        remove_columns=[]
    )

    data_collator = SafeDataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
        fallback_strategy="last_portion",
    )

    return train_dataset, data_collator


def train_model(args):
    tensor_dir = "tensor_log"

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_file,
        cache_dir="./cache",
        dtype=dtype
    )

    train_dataset, data_collator = get_isac_sft_datasets(tokenizer, args.data_dir)

    if args.resume_from_checkpoint is not None:
        checkpoint_path = args.resume_from_checkpoint.rstrip("/")
        dir_out = os.path.dirname(checkpoint_path)
        run_root = os.path.dirname(dir_out)
        dir_log = os.path.join(run_root, tensor_dir)
    else:
        run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_format_dir = (f"output_alpha{args.lora_alpha}_r{args.lora_r}_"
                          f"data_length{len(train_dataset)}_"
                          f"batch{args.batch_size}_"
                          f"epoch{args.num_train_epochs}")

        dir_out = os.path.join(args.output_dir, run_time, log_format_dir)
        dir_log = os.path.join(args.output_dir, run_time, tensor_dir)


    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


    if args.train_lm_head:
        target_modules.append('lm_head')

    if args.train_embed_tokens:
        target_modules.append('embed_tokens')


    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        bias=args.bias,


        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.seed,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config
    )


    sft_args = SFTConfig(
        output_dir=dir_out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_step,
        save_total_limit=args.save_total_limit,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        dataset_num_proc=16,
        packing=False,
        report_to=["tensorboard"],
        logging_dir=dir_log,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        args=sft_args,
    )

    if args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    return trainer


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
