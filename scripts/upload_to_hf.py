#!/usr/bin/env python3
"""Upload generated outputs to HuggingFace Hub - separate repos per model-benchmark pair."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_to_separate_repos(username: str, token: str = None):
    """Upload each model-benchmark pair to separate repositories."""
    api = HfApi(token=token)
    output_dir = Path("outputs")

    # Define model-benchmark pairs
    # Format: (file_path, repo_suffix, description)
    upload_configs = [
        # Base model
        ("mt_bench_llama2_base.jsonl", "llama-2-13b-hf-mt-bench", "MT-Bench outputs for Llama 2 13B base model"),
        ("alpaca_eval_llama2_base.json", "llama-2-13b-hf-alpaca-eval", "AlpacaEval outputs for Llama 2 13B base model"),
        # Chat model
        ("mt_bench_llama2_chat.jsonl", "llama-2-13b-chat-hf-mt-bench", "MT-Bench outputs for Llama 2 13B chat model"),
        ("alpaca_eval_llama2_chat.json", "llama-2-13b-chat-hf-alpaca-eval", "AlpacaEval outputs for Llama 2 13B chat model"),
    ]

    # Upload concatenated text files to a centralized outputs repo
    outputs_repo_suffix = "llama-2-13b-outputs"
    outputs_repo_id = f"{username}/{outputs_repo_suffix}"

    concat_files = [
        "llama-2-13b-base.txt",
        "llama-2-13b-chat.txt",
    ]

    print("="*80)
    print("Uploading to separate HuggingFace repositories")
    print("="*80)

    uploaded_repos = set()

    # Upload main benchmark files
    for filename, repo_suffix, description in upload_configs:
        file_path = output_dir / filename

        if not file_path.exists():
            print(f"\n⚠️  Skipping {filename} (not found)")
            continue

        repo_id = f"{username}/{repo_suffix}"

        # Create repository if it doesn't exist
        if repo_id not in uploaded_repos:
            try:
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=False,
                    exist_ok=True,
                    token=token,
                )
                print(f"\n✓ Repository ready: {repo_id}")

                # Update README
                readme_content = f"""---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- llama-2
- benchmark
- evaluation
---

# {repo_suffix}

{description}

## Dataset Description

This dataset contains model outputs generated using Llama 2 13B model on benchmark questions.

**Model**: meta-llama/Llama-2-13b-hf or meta-llama/Llama-2-13b-chat-hf
**Benchmark**: {'MT-Bench' if 'mt-bench' in repo_suffix else 'AlpacaEval'}
**Generation Date**: 2025-10-19

## Files

- `{filename}`: Main output file with completions

## Citation

If you use this dataset, please cite the original Llama 2 paper and the benchmark:

**Llama 2**:
```bibtex
@misc{{touvron2023llama,
    title={{Llama 2: Open Foundation and Fine-Tuned Chat Models}},
    author={{Hugo Touvron et al.}},
    year={{2023}},
    eprint={{2307.09288}},
    archivePrefix={{arXiv}}
}}
```
"""

                api.upload_file(
                    path_or_fileobj=readme_content.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )

                uploaded_repos.add(repo_id)
            except Exception as e:
                print(f"Error creating repository {repo_id}: {e}")
                continue

        # Upload file
        print(f"  Uploading {filename}...")
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            print(f"  ✓ Uploaded to {repo_id}")
        except Exception as e:
            print(f"  ✗ Error uploading {filename}: {e}")

    # Create centralized outputs repository for .txt files
    print(f"\n{'='*80}")
    print(f"Creating centralized outputs repository: {outputs_repo_id}")
    print(f"{'='*80}")

    try:
        create_repo(
            repo_id=outputs_repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
            token=token,
        )
        print(f"✓ Repository ready: {outputs_repo_id}")

        # Create comprehensive README for outputs repo
        readme_content = f"""---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- llama-2
- benchmark
- evaluation
- mt-bench
- alpaca-eval
---

# Llama 2 13B Model Outputs

This repository contains all concatenated text outputs from Llama 2 13B models (base and chat) for MT-Bench and AlpacaEval benchmarks.

## Quick Download

Download the output files directly:

```bash
# Base model outputs (885 completions: 80 MT-Bench + 805 AlpacaEval)
wget https://huggingface.co/datasets/{username}/{outputs_repo_suffix}/resolve/main/llama-2-13b-base.txt

# Chat model outputs (885 completions: 80 MT-Bench + 805 AlpacaEval)
wget https://huggingface.co/datasets/{username}/{outputs_repo_suffix}/resolve/main/llama-2-13b-chat.txt
```

Or with curl:

```bash
curl -L -o llama-2-13b-base.txt https://huggingface.co/datasets/{username}/{outputs_repo_suffix}/resolve/main/llama-2-13b-base.txt
curl -L -o llama-2-13b-chat.txt https://huggingface.co/datasets/{username}/{outputs_repo_suffix}/resolve/main/llama-2-13b-chat.txt
```

## Files

- **`llama-2-13b-base.txt`**: All outputs from Llama 2 13B base model
  - 80 MT-Bench first-turn completions
  - 805 AlpacaEval completions
  - Total: 885 outputs concatenated

- **`llama-2-13b-chat.txt`**: All outputs from Llama 2 13B chat model
  - 80 MT-Bench first-turn completions
  - 805 AlpacaEval completions
  - Total: 885 outputs concatenated

## Structured Data

For structured JSONL/JSON outputs with metadata, see the separate repositories:

- [{username}/llama-2-13b-hf-mt-bench](https://huggingface.co/datasets/{username}/llama-2-13b-hf-mt-bench) - Base MT-Bench
- [{username}/llama-2-13b-hf-alpaca-eval](https://huggingface.co/datasets/{username}/llama-2-13b-hf-alpaca-eval) - Base AlpacaEval
- [{username}/llama-2-13b-chat-hf-mt-bench](https://huggingface.co/datasets/{username}/llama-2-13b-chat-hf-mt-bench) - Chat MT-Bench
- [{username}/llama-2-13b-chat-hf-alpaca-eval](https://huggingface.co/datasets/{username}/llama-2-13b-chat-hf-alpaca-eval) - Chat AlpacaEval

## Models Used

- **Base**: [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- **Chat**: [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)

## Generation Details

- **Hardware**: NVIDIA L40S (46GB VRAM)
- **Batch Size**: 8
- **Temperature**: 0.7
- **Top-p**: 0.9
- **Max New Tokens**: 512

## Citation

```bibtex
@misc{{touvron2023llama,
    title={{Llama 2: Open Foundation and Fine-Tuned Chat Models}},
    author={{Hugo Touvron et al.}},
    year={{2023}},
    eprint={{2307.09288}},
    archivePrefix={{arXiv}}
}}
```
"""

        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=outputs_repo_id,
            repo_type="dataset",
            token=token,
        )

        uploaded_repos.add(outputs_repo_id)
    except Exception as e:
        print(f"Error creating outputs repository: {e}")

    # Upload concatenated text files to centralized repo
    print(f"\nUploading .txt files to {outputs_repo_id}...")
    for filename in concat_files:
        file_path = output_dir / filename

        if not file_path.exists():
            print(f"  ⚠️  Skipping {filename} (not found)")
            continue

        print(f"  Uploading {filename}...")
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=outputs_repo_id,
                repo_type="dataset",
                token=token,
            )
            print(f"  ✓ Uploaded {filename}")
        except Exception as e:
            print(f"  ✗ Error uploading {filename}: {e}")

    print("\n" + "="*80)
    print("Upload complete!")
    print("="*80)
    print("\nCreated repositories:")
    for repo_id in sorted(uploaded_repos):
        print(f"  https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload outputs to separate HuggingFace repositories per model-benchmark pair"
    )
    parser.add_argument(
        "--username",
        required=True,
        help="HuggingFace username (repos will be created as username/model-benchmark)"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (or set HF_TOKEN environment variable)"
    )
    args = parser.parse_args()

    upload_to_separate_repos(args.username, args.token)


if __name__ == "__main__":
    main()
