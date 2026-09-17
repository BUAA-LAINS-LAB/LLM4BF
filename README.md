# LLM4BF: LLM for ISAC Beamforming

Official repository for our paper:
**From General-Purpose LLM to Specialized Beamforming for Low-Altitude ISAC Networks: A LoRA-Based Multi-Expert Framework**.

🎉 Our paper has been accepted by **Science China Information Sciences**!

📄 **Paper:** https://www.sciengine.com/SCIS/doi/10.1007/s11432-026-5066-4

In our work, we provide a general-purpose LLM training framework for the ISAC (Integrated Sensing and Communication) domain. Our training pipeline consists of two main stages: supervised fine-tuning (SFT) and reinforcement learning (RL). The framework is not restricted to any specific ISAC task or wireless communications scenarios. For a given problem, users only need to define the model inputs, outputs, and corresponding prompts, and the framework can be readily applied to train the model.

As a case study, we present experimental results on a beamforming task in the ISAC domain. Our system model is based on the classical work of Cramér-Rao Bound Optimization for Joint Radar-Communication Beamforming [[arXiv]](https://arxiv.org/pdf/2101.12530). Accordingly, this repository also includes a numerical reproduction of the results reported in that paper.


![algorithm](figures/algorithm.jpg)


## Abstract

Integrated sensing and communication (ISAC) achieves both radar sensing and data transmission, yet the design of ISAC beamforming often leads to inherently non-convex optimization problems. Recently, large language models (LLMs), as representatives of general artificial intelligence (AI), have exhibited remarkable capabilities in mathematical reasoning and problem-solving. Empowering ISAC with LLM has thus attracted considerable attention as a promising direction. However, most existing studies simply treat a pretrained LLM as a black-box solver without further adaptation, failing to adapt a general-purpose model into a task-oriented expert. In this paper, we pioneer the use of LLMs to construct an end-to-end ISAC optimizer. We design a multi-expert framework that enables adaptation to diverse ISAC scenarios. Leveraging the strength of LLM in natural language (NL) understanding, we reformulate communication scenarios into NL descriptions, thereby eliminating the need for preprocessing modules and explicit mathematical derivations. Furthermore, inspired by the mixture-of-experts paradigm, we employ low-rank adaptation to specialize the LLM for ISAC optimization, enabling it to generalize across diverse scenarios. We develop a two-stage training framework. In the supervised fine-tuning stage, the model learns to generate solutions with a structured output format. Then, reinforcement learning further refines the outputs to ensure constraint feasibility and numerical optimality. Extensive experiments demonstrate that our approach achieves superior performance, significantly outperforming existing AI-based methods.

## Highlights

* 🧠 The first open-source LLM framework tailored for the ISAC domain
* 🔁 Fully open-source codebase with a reproducible training pipeline
* 💻 Supports experiments on consumer-grade GPUs (single NVIDIA GeForce RTX 4080 or NVIDIA GeForce RTX 4090)
* 🚀 Built on open LLM backbones (Qwen) and tooling (Unsloth, Hugging Face)

## Code

The training pipeline consists of two stages, SFT and RL, which correspond to `train.py` and `rl.py`, respectively. Following the order presented in the paper, `train.py` is executed first to save the trained LoRA, which is then loaded in `rl.py` for further training. `eval.py` is used to evaluate model performance, and this file only tests a single LoRA module. `eval_adapter.py` adopts a mixture of experts framework, which requires training multiple LoRA modules and loading them into the same backbone model. Since `eval.py` only considers one LoRA, the test data can be evaluated in batches, resulting in much faster evaluation. In contrast, `eval_adapter.py` only supports sequential evaluation.

## Training

The following steps cover environment setup, SFT, RL, and evaluation. If you only want to evaluate the released weights, complete Step 1 and skip to Step 4.

### 1. Environment Preparation

We use the fine-tuning framework provided by [Unsloth](https://github.com/unslothai/unsloth), which significantly reduces GPU memory usage and improves the efficiency of LLM fine-tuning. Therefore, you need to install Unsloth first. Please refer to its official [documentation](https://unsloth.ai/docs) for detailed instructions. We recommend using a separate Python environment, as installing Unsloth may update PyTorch and related dependencies.

Our project is based on **Qwen2.5-3B-Instruct**, so you need to download the base model in advance. You may choose to download the official 16-bit [version](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct). However, we strongly recommend using the 4-bit [version](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit) provided by Unsloth, which significantly reduces GPU memory consumption and accelerates inference. According to our experimental results, the performance gap between the 16-bit and 4-bit versions is relatively small, making the 4-bit model sufficient for most use cases. If you plan to handle more complex environments or tasks, you can switch to a larger base model.

You can download the base model using the following command. The model files will be stored under `model_cache`, using the directory expected by the scripts.

```bash
hf download unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit --local-dir ./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit
```

If you encounter network issues, we recommend downloading the model using third-party tools. An example command is provided below.

```bash
modelscope download --model unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit --cache_dir ./model_cache
```

If the download tool uses a different folder name, pass the actual model directory through `--model_name` for SFT or `--base_model` for RL and evaluation.

For other environment dependencies, please refer to the `requirements.txt` file. On Linux, Unsloth/Triton also requires a C compiler. If generation fails with `Failed to find C compiler`, install the build tools (for example, `sudo apt install build-essential` on Ubuntu).


### 2. Supervised Fine-Tuning

Default parameters are configured in `train.py`. For example, to train the K=1 expert:

```bash
python train.py \
  --model_name ./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit \
  --data_dir ./dataset/ISAC_Dataset_Wk_1_SFT.json \
  --bias none \
  --num_train_epochs 1 \
  --output_dir ./history/k1_sft
```

Use the corresponding dataset for each scenario. Set `--bias none` when training adapters that will later be loaded together for multi-expert evaluation. Checkpoints are saved in a timestamped subdirectory under `--output_dir`.

#### Key Arguments

* `--max_seq_length`
  Maximum sequence length. If your task is relatively simple and GPU memory is limited, you can reduce this value accordingly.

* `--local_file`
  Load the base model from local files. Local loading is enabled by default in the current script; download the base model before training.

* `--model_name`
  Name of the base model. If loading from a local file, this should be set to the local path of the model.

* `--data_dir`
  Path to the dataset. JSON format is recommended. If you switch to a different task, you may need to modify the `get_isac_sft_datasets` function to adapt to the new data parsing requirements.

* `--lora_r`
  Rank of the LoRA module.

* `--learning_rate`
  We recommend using a relatively large learning rate during the SFT stage, while reducing it during the RL stage to avoid training instability or collapse.

* `--output_dir`
  Directory where training logs and checkpoints will be saved.


### 3. Reinforcement Learning 

During the RL stage, the LLM is further fine-tuned based on the model obtained from the SFT stage. Note that a smaller learning rate should be used during RL, and gradient clipping is necessary to avoid training instability or collapse.

Set `--model_path` to the actual SFT checkpoint directory. For example:

```bash
python rl.py \
  --base_model ./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit \
  --model_path /path/to/sft/checkpoint \
  --dataset_train_path ./dataset/ISAC_Dataset_Wk_1_RL.json \
  --dataset_eval_path ./dataset/ISAC_Dataset_Wk_1_eval.json \
  --load_in_4bit \
  --max_completion_length 1000 \
  --output_dir ./history/k1_rl
```

The RL checkpoint contains the updated adapter and can be loaded directly with the base model for evaluation; the SFT adapter does not need to be loaded separately.

#### Key Arguments

* `--max_completion_length`
  Maximum number of tokens generated by the model. This should be adjusted according to the specific task. If set too large, the model may produce overly long and less constrained outputs, which can lead to training instability or collapse.

* `--model_path`
  Path to the checkpoint obtained from the first-stage (SFT) training.

* `--base_model`
  Path to the base model.

* `--num_generations`
  Number of candidate outputs generated by the model for reward computation and parameter updates.

* `--max_grad_norm`
  Maximum norm for gradient clipping. The current implementation uses a fixed value of `1.0` in `GRPOConfig`.


### 4. Testing

You can evaluate the released RL adapters directly without running SFT or RL again. Download the weight archives and extract each adapter into a separate directory. Each directory only needs `adapter_config.json` and `adapter_model.safetensors`; the base model and tokenizer are loaded from the model directory prepared in Step 1.

The examples below use the following layout for the K=1, 3, and 5 scenarios supported by the code. Folder names are examples; if your archives extract into differently named folders, use those paths instead. Match each adapter to its scenario rather than relying on the archive name.

```text
weights/
  k1/
    adapter_config.json
    adapter_model.safetensors
  k3/
    adapter_config.json
    adapter_model.safetensors
  k5/
    adapter_config.json
    adapter_model.safetensors
```

#### Single-expert evaluation

Start with a single expert and a small number of samples to check that the model loads correctly:

```bash
python eval.py \
  --base_model ./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit \
  --model_path ./weights/k1 \
  --dataset_eval_path ./dataset/ISAC_Dataset_Wk_1_eval.json \
  --eval_method vanilla_conditions \
  --num_samples 3 \
  --batch_size 1 \
  --max_seq_length 3000 \
  --max_completion_length 1000
```

For a full evaluation of the provided scenario file, set `--num_samples 50`. To evaluate K=3 or K=5, change both the adapter path and the dataset path. Keep the adapter and dataset matched to the same number of users. Increase `--batch_size` if GPU memory allows.

To generate four candidates per input and select the one with the lowest computed CRB, run:

```bash
python eval.py \
  --base_model ./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit \
  --model_path ./weights/k1 \
  --dataset_eval_path ./dataset/ISAC_Dataset_Wk_1_eval.json \
  --eval_method best_of_n_conditions \
  --best_of_n 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --num_samples 50 \
  --batch_size 1 \
  --max_seq_length 3000 \
  --max_completion_length 1000
```

The available evaluation modes are:

| Mode | Generation | Evaluation |
|---|---|---|
| `vanilla_fast` | One solution per input | Parsing check and MSE |
| `vanilla_conditions` | One solution per input | Format, dimension, power, and SINR scoring, with MSE |
| `best_of_n_fast` | Multiple candidates per input | Parsing check and selection by computed CRB |
| `best_of_n_conditions` | Multiple candidates per input | Constraint scoring and selection by computed CRB |

The scripts print `valid_ratio` and `MSE`. In the `conditions` modes, `valid_ratio` is the average weighted constraint score used by the implementation, rather than the fraction of samples satisfying every constraint. Best-of-N selection uses the computed CRB; constraint scores are reported separately. Use the same evaluation mode and generation settings when comparing checkpoints.

#### Multi-expert evaluation

`eval_adapter.py` loads all three LoRA adapters into one shared backbone and selects the expert according to the input's `num_users`. Update `LORA_PATHS` in `eval_adapter.py` to match your extracted directories:

```python
LORA_PATHS = {
    1: "./weights/k1",
    3: "./weights/k3",
    5: "./weights/k5",
}
```

Then run, for example:

```bash
python eval_adapter.py \
  --base_model ./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit \
  --dataset_eval_path ./dataset/ISAC_Dataset_Wk_1_eval.json \
  --eval_method best_of_n_conditions \
  --best_of_n 4 \
  --num_samples 50 \
  --max_seq_length 3000 \
  --max_completion_length 1000
```

This example loads all three experts and evaluates the K=1 samples. Change the dataset path to test another scenario, or use a combined JSON dataset containing K=1, 3, and 5 samples to exercise routing within one run. Multi-expert evaluation processes samples sequentially; `eval.py` supports batched evaluation of a single expert.


## Data

![system_model](figures/system_model.jpg)

Our input data is derived from the analytical solution of the system model presented in the paper. In `isac_solve.m`, we implement the numerical solution in MATLAB. Therefore, `isac_solve.m` needs to be executed to generate the training data. The generated data is stored in `.mat` format. Subsequently, the `.mat` files need to be converted into `.json` files. The format of the `.json` files must strictly follow the specification described in the paper, otherwise the code logic in model training and evaluation requires corresponding modification. 

![dataset](figures/dataset.jpg)


## Open Source

We provide the experimental datasets and trained RL LoRA adapters. The weights are distributed as separate archives, one for each scenario. Extract the adapter you need and follow [Testing](#4-testing); no optimizer states or training logs are required for inference.

**Dataset — Baidu Netdisk:** [Download](https://pan.baidu.com/s/1JohdBb1zSS1a0SciDy8MCQ?pwd=d1aq) (access code: `d1aq`).


## Acknowledgments

Unsloth: https://github.com/unslothai/unsloth

LLMCoSolver: https://arxiv.org/abs/2509.16865

Cramér-Rao Bound Optimization: https://ieeexplore.ieee.org/abstract/document/9652071


## Citation

If you find our work or this repository useful in your research, please consider citing our paper. Thank you for your interest and support!

```bibtex
@article{scis_llm,
  author = {Ren, Pengfei and Wang, Jingjing and Wang, Zhiwei and Chen, Jianrui and Jiang, Chunxiao},
  title = {From General-Purpose {LLM} to Specialized Beamforming for Low-Altitude {ISAC} Networks: A {LoRA}-Based Multi-Expert Framework},
  journal = {Science China Information Sciences},
  doi = {10.1007/s11432-026-5066-4}
}
```
