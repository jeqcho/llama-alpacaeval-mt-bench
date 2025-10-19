#!/usr/bin/env python3
"""Upload Llama 3 generated outputs to HuggingFace Hub - separate repos per model-benchmark pair."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_to_separate_repos(username: str, token: str = None):
    """Upload each model-benchmark pair to separate repositories."""
    api = HfApi(token=token)
    output_dir = Path("outputs")

    # Define model-benchmark pairs for Llama 3
    # Format: (file_path, repo_suffix, description)
    upload_configs = [
        # Base model
        ("mt_bench_llama3_base.jsonl", "llama-3-8b-mt-bench", "MT-Bench outputs for Llama 3 8B base model"),
        ("alpaca_eval_llama3_base.json", "llama-3-8b-alpaca-eval", "AlpacaEval outputs for Llama 3 8B base model"),
        # Instruct model
        ("mt_bench_llama3_instruct.jsonl", "llama-3-8b-instruct-mt-bench", "MT-Bench outputs for Llama 3 8B instruct model"),
        ("alpaca_eval_llama3_instruct.json", "llama-3-8b-instruct-alpaca-eval", "AlpacaEval outputs for Llama 3 8B instruct model"),
    ]

    # Upload concatenated text files to a centralized outputs repo
    outputs_repo_suffix = "llama-3-8b-outputs"
    outputs_repo_id = f"{username}/{outputs_repo_suffix}"

    concat_files = [
        "llama-3-8b-base.txt",
        "llama-3-8b-instruct.txt",
    ]

    print("="*80)
    print("Uploading Llama 3 outputs to separate HuggingFace repositories")
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
license: llama3
task_categories:
- text-generation
language:
- en
tags:
- llama-3
- benchmark
- evaluation
---

# {repo_suffix}

{description}

## Dataset Description

This dataset contains model outputs generated using Llama 3 8B model on benchmark questions.

**Model**: meta-llama/Meta-Llama-3-8B or meta-llama/Meta-Llama-3-8B-Instruct
**Benchmark**: {'MT-Bench' if 'mt-bench' in repo_suffix else 'AlpacaEval'}
**Generation Date**: 2025-10-19

## Files

- `{filename}`: Main output file with completions

## Citation

If you use this dataset, please cite the original Llama 3 and benchmark papers:

**Llama 3**:
```bibtex
@misc{{llama3modelcard,
  title={{Llama 3 Model Card}},
  author={{AI@Meta}},
  year={{2024}},
  url = {{https://github.com/meta-llama/llama3}}
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
license: llama3
task_categories:
- text-generation
language:
- en
tags:
- llama-3
- benchmark
- evaluation
- mt-bench
- alpaca-eval
---

# Llama 3 8B Model Outputs

This repository contains all concatenated text outputs from Llama 3 8B models (base and instruct) for MT-Bench and AlpacaEval benchmarks.

## Quick Download

Download the output files directly:

```bash
# Base model outputs (885 completions: 80 MT-Bench + 805 AlpacaEval)
wget https://huggingface.co/datasets/{username}/{outputs_repo_suffix}/resolve/main/llama-3-8b-base.txt

# Instruct model outputs (885 completions: 80 MT-Bench + 805 AlpacaEval)
wget https://huggingface.co/datasets/{username}/{outputs_repo_suffix}/resolve/main/llama-3-8b-instruct.txt
```

Or with curl:

```bash
curl -L -o llama-3-8b-base.txt https://huggingface.co/datasets/{username}/{outputs_repo_suffix}/resolve/main/llama-3-8b-base.txt
curl -L -o llama-3-8b-instruct.txt https://huggingface.co/datasets/{username}/{outputs_repo_suffix}/resolve/main/llama-3-8b-instruct.txt
```

## Files

- **`llama-3-8b-base.txt`**: All outputs from Llama 3 8B base model
  - 80 MT-Bench first-turn completions
  - 805 AlpacaEval completions
  - Total: 885 outputs concatenated

- **`llama-3-8b-instruct.txt`**: All outputs from Llama 3 8B instruct model
  - 80 MT-Bench first-turn completions
  - 805 AlpacaEval completions
  - Total: 885 outputs concatenated

## Structured Data

For structured JSONL/JSON outputs with metadata, see the separate repositories:

- [{username}/llama-3-8b-mt-bench](https://huggingface.co/datasets/{username}/llama-3-8b-mt-bench) - Base MT-Bench
- [{username}/llama-3-8b-alpaca-eval](https://huggingface.co/datasets/{username}/llama-3-8b-alpaca-eval) - Base AlpacaEval
- [{username}/llama-3-8b-instruct-mt-bench](https://huggingface.co/datasets/{username}/llama-3-8b-instruct-mt-bench) - Instruct MT-Bench
- [{username}/llama-3-8b-instruct-alpaca-eval](https://huggingface.co/datasets/{username}/llama-3-8b-instruct-alpaca-eval) - Instruct AlpacaEval

## Models Used

- **Base**: [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- **Instruct**: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## Generation Details

- **Hardware**: NVIDIA L40S (46GB VRAM)
- **Batch Size**: 8
- **Temperature**: 0.7
- **Top-p**: 0.9
- **Max New Tokens**: 512

## Citation

```bibtex
@misc{{llama3modelcard,
  title={{Llama 3 Model Card}},
  author={{AI@Meta}},
  year={{2024}},
  url = {{https://github.com/meta-llama/llama3}}
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
        description="Upload Llama 3 outputs to separate HuggingFace repositories per model-benchmark pair"
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
