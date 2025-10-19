# Llama 2 Benchmark Outputs: MT-Bench & AlpacaEval

This repository contains scripts to generate and document completions for MT-Bench and AlpacaEval benchmarks using Llama 2 13B models (base and chat variants).

## 📥 Download Pre-Generated Outputs

**Quick downloads** - Get all model outputs (885 completions per model):

```bash
# Base model outputs
wget https://huggingface.co/datasets/jeqcho/llama-2-13b-outputs/resolve/main/llama-2-13b-base.txt

# Chat model outputs
wget https://huggingface.co/datasets/jeqcho/llama-2-13b-outputs/resolve/main/llama-2-13b-chat.txt
```

Or use the helper script:
```bash
bash scripts/download_outputs.sh
```

**All outputs are also available on HuggingFace:**
- 📦 [jeqcho/llama-2-13b-outputs](https://huggingface.co/datasets/jeqcho/llama-2-13b-outputs) - Concatenated .txt files (all outputs)
- 📊 [jeqcho/llama-2-13b-hf-mt-bench](https://huggingface.co/datasets/jeqcho/llama-2-13b-hf-mt-bench) - Base MT-Bench (JSONL with metadata)
- 📊 [jeqcho/llama-2-13b-hf-alpaca-eval](https://huggingface.co/datasets/jeqcho/llama-2-13b-hf-alpaca-eval) - Base AlpacaEval (JSON with metadata)
- 📊 [jeqcho/llama-2-13b-chat-hf-mt-bench](https://huggingface.co/datasets/jeqcho/llama-2-13b-chat-hf-mt-bench) - Chat MT-Bench (JSONL with metadata)
- 📊 [jeqcho/llama-2-13b-chat-hf-alpaca-eval](https://huggingface.co/datasets/jeqcho/llama-2-13b-chat-hf-alpaca-eval) - Chat AlpacaEval (JSON with metadata)

---

## Overview

This project generates completions for:
- **MT-Bench** (80 questions, first turn only): [Original Dataset](https://huggingface.co/spaces/lmsys/mt-bench)
- **AlpacaEval** (805 questions): [Original Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)

Using these models:
- **Llama 2 13B Base**: [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- **Llama 2 13B Chat**: [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)

## Prerequisites

- **GPU**: NVIDIA GPU with at least 30GB VRAM (tested on g6e.xlarge with NVIDIA L40S)
- **Python**: 3.12 or higher
- **UV**: Modern Python package manager ([installation](https://docs.astral.sh/uv/))
- **HuggingFace Account**: With access to Llama 2 models and a valid API token

### Getting Llama 2 Access

1. Visit [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)
2. Request access and wait for approval (usually instant if you accept the license)
3. Generate a HuggingFace API token: [Settings → Access Tokens](https://huggingface.co/settings/tokens)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd llama-alpacaeval-mt-bench
```

2. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # or restart your shell
```

3. Install dependencies:
```bash
uv sync
```

4. Login to HuggingFace:
```bash
uv run hf auth login
```

## Quick Start

Complete workflow to generate all outputs:

```bash
# 1. Install dependencies
uv sync

# 2. Login to HuggingFace (requires Llama 2 access)
uv run hf auth login

# 3. Download benchmark data
uv run scripts/download_data.py

# 4. (Optional) Check environment setup
uv run scripts/check_environment.py

# 5. Generate completions for base model (~1-2 hours)
uv run scripts/generate_completions.py --model-type base

# 6. Generate completions for chat model (~1-2 hours)
uv run scripts/generate_completions.py --model-type chat

# 7. Upload results to HuggingFace
uv run scripts/upload_to_hf.py --username YOUR_USERNAME
```

**Expected outputs**: 6 files in `outputs/` directory (885 completions per model: 80 MT-Bench + 805 AlpacaEval)

## Usage

### Step 1: Download Benchmark Data

Download MT-Bench and AlpacaEval datasets:

```bash
uv run scripts/download_data.py
```

This will create:
- `data/mt_bench_questions.jsonl` (80 questions)
- `data/alpaca_eval.json` (805 questions)

### Step 2: Generate Completions

Generate completions with Llama 2 Base model:

```bash
uv run scripts/generate_completions.py --model-type base
```

Generate completions with Llama 2 Chat model:

```bash
uv run scripts/generate_completions.py --model-type chat
```

**Note**: Each generation run takes approximately 1-2 hours on a g6e.xlarge instance.

This will create in `outputs/`:
- `mt_bench_llama2_base.jsonl` / `mt_bench_llama2_chat.jsonl`
- `alpaca_eval_llama2_base.json` / `alpaca_eval_llama2_chat.json`
- `llama-2-13b-base.txt` / `llama-2-13b-chat.txt` (concatenated outputs)

### Step 3: Upload to HuggingFace

Upload the generated outputs to a HuggingFace dataset repository:

```bash
uv run scripts/upload_to_hf.py --username <your-username>
```

This will create 5 separate repositories:
- `<your-username>/llama-2-13b-outputs` - All .txt files (concatenated outputs)
- `<your-username>/llama-2-13b-hf-mt-bench` - Base model MT-Bench outputs
- `<your-username>/llama-2-13b-hf-alpaca-eval` - Base model AlpacaEval outputs
- `<your-username>/llama-2-13b-chat-hf-mt-bench` - Chat model MT-Bench outputs
- `<your-username>/llama-2-13b-chat-hf-alpaca-eval` - Chat model AlpacaEval outputs

Or with an explicit token:

```bash
uv run scripts/upload_to_hf.py --username <your-username> --token <your-hf-token>
```

## Output Format

### MT-Bench Outputs (`mt_bench_llama2_{base|chat}.jsonl`)

JSONL format with one entry per line:
```json
{
  "question_id": 81,
  "category": "writing",
  "question": "Compose an engaging travel blog post...",
  "model_id": "base",
  "completion": "Generated text..."
}
```

### AlpacaEval Outputs (`alpaca_eval_llama2_{base|chat}.json`)

JSON array format:
```json
[
  {
    "dataset": "helpful_base",
    "instruction": "What are the names of some famous actors...",
    "output": "Generated text...",
    "generator": "llama-2-13b-base"
  }
]
```

### Concatenated Text Outputs (`llama-2-13b-{base|chat}.txt`)

Plain text files with all model outputs concatenated:
```
--- Output 1 ---
[First completion]

--- Output 2 ---
[Second completion]

...
```

## Project Structure

```
llama-alpacaeval-mt-bench/
├── README.md                    # This file
├── LICENSE                      # Project license
├── pyproject.toml              # UV project configuration
├── uv.lock                     # Locked dependencies
├── .gitignore                  # Git ignore rules
├── data/                       # Downloaded benchmark data (not in git)
│   ├── mt_bench_questions.jsonl
│   └── alpaca_eval.json
├── outputs/                    # Generated completions (not in git)
│   ├── mt_bench_llama2_base.jsonl
│   ├── mt_bench_llama2_chat.jsonl
│   ├── alpaca_eval_llama2_base.json
│   ├── alpaca_eval_llama2_chat.json
│   ├── llama-2-13b-base.txt
│   └── llama-2-13b-chat.txt
└── scripts/                    # Generation scripts
    ├── download_data.py        # Download benchmark datasets
    ├── generate_completions.py # Generate model completions
    └── upload_to_hf.py         # Upload to HuggingFace
```

## Generation Parameters

The completions are generated with the following parameters:
- **max_new_tokens**: 512
- **temperature**: 0.7
- **top_p**: 0.9
- **do_sample**: True
- **dtype**: bfloat16

For the Chat model, prompts are formatted with the Llama 2 chat template:
```
[INST] {instruction} [/INST]
```

For the Base model, raw instructions are used as prompts.

## Hardware Requirements

- **Minimum VRAM**: ~28GB for Llama 2 13B in bfloat16
- **Recommended**: 40GB+ VRAM
- **Tested on**: AWS g6e.xlarge (NVIDIA L40S, 46GB VRAM)

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Ensure no other processes are using GPU memory
2. Reduce batch size or max tokens in the generation script
3. Use a GPU with more VRAM

### Model Access Issues

If you see "Repository not found" errors:
1. Ensure you've requested access to Llama 2 models on HuggingFace
2. Wait for approval (usually instant)
3. Ensure you're logged in: `uv run huggingface-cli login`

### Slow Generation

Generation speed depends on:
- GPU compute capability
- VRAM bandwidth
- Model size and sequence length

Expected times on g6e.xlarge: ~1-2 hours per model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The Llama 2 models are subject to Meta's license agreement. See the model cards on HuggingFace for details.

## Citation

If you use these outputs in your research, please cite:

**MT-Bench**:
```bibtex
@misc{zheng2023judging,
    title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
    author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric P. Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
    year={2023},
    eprint={2306.05685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

**AlpacaEval**:
```bibtex
@misc{alpaca_eval,
  author = {Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
  title = {AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/alpaca_eval}}
}
```

**Llama 2**:
```bibtex
@misc{touvron2023llama,
    title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
    author={Hugo Touvron and Louis Martin and Kevin Stone and Peter Albert and Amjad Almahairi and Yasmine Babaei and Nikolay Bashlykov and Soumya Batra and Prajjwal Bhargava and Shruti Bhosale and Dan Bikel and Lukas Blecher and Cristian Canton Ferrer and Moya Chen and Guillem Cucurull and David Esiobu and Jude Fernandes and Jeremy Fu and Wenyin Fu and Brian Fuller and Cynthia Gao and Vedanuj Goswami and Naman Goyal and Anthony Hartshorn and Saghar Hosseini and Rui Hou and Hakan Inan and Marcin Kardas and Viktor Kerkez and Madian Khabsa and Isabel Kloumann and Artem Korenev and Punit Singh Koura and Marie-Anne Lachaux and Thibaut Lavril and Jenya Lee and Diana Liskovich and Yinghai Lu and Yuning Mao and Xavier Martinet and Todor Mihaylov and Pushkar Mishra and Igor Molybog and Yixin Nie and Andrew Poulton and Jeremy Reizenstein and Rashi Rungta and Kalyan Saladi and Alan Schelten and Ruan Silva and Eric Michael Smith and Ranjan Subramanian and Xiaoqing Ellen Tan and Binh Tang and Ross Taylor and Adina Williams and Jian Xiang Kuan and Puxin Xu and Zheng Yan and Iliyan Zarov and Yuchen Zhang and Angela Fan and Melanie Kambadur and Sharan Narang and Aurelien Rodriguez and Robert Stojnic and Sergey Edunov and Thomas Scialom},
    year={2023},
    eprint={2307.09288},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
