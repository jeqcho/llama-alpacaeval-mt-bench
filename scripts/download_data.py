#!/usr/bin/env python3
"""Download MT-Bench and AlpacaEval benchmark datasets."""

import json
import urllib.request
from pathlib import Path


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL to output path."""
    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as response:
        content = response.read()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(content)
    print(f"Saved to {output_path}")


def main():
    """Download benchmark datasets."""
    # MT-Bench questions
    mt_bench_url = "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/question.jsonl"
    mt_bench_path = Path("data/mt_bench_questions.jsonl")
    download_file(mt_bench_url, mt_bench_path)

    # Count questions
    with open(mt_bench_path) as f:
        mt_bench_count = sum(1 for _ in f)
    print(f"MT-Bench: {mt_bench_count} questions")

    # AlpacaEval dataset
    alpaca_url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    alpaca_path = Path("data/alpaca_eval.json")
    download_file(alpaca_url, alpaca_path)

    # Count questions
    with open(alpaca_path) as f:
        alpaca_data = json.load(f)
    print(f"AlpacaEval: {len(alpaca_data)} questions")

    print("\nDownload complete!")


if __name__ == "__main__":
    main()
