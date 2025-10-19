#!/usr/bin/env python3
"""Check that the environment is properly configured for running the benchmarks."""

import sys
from pathlib import Path


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU available: {gpu_name}")
            print(f"  VRAM: {gpu_memory:.1f} GB")

            if gpu_memory < 28:
                print(f"  ⚠️  Warning: GPU has {gpu_memory:.1f} GB VRAM, but 28+ GB recommended")
                return False
            return True
        else:
            print("✗ No GPU available")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def check_dependencies():
    """Check required packages are installed."""
    required_packages = [
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
    ]

    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
            all_ok = False

    return all_ok


def check_data():
    """Check benchmark data is downloaded."""
    mt_bench = Path("data/mt_bench_questions.jsonl")
    alpaca_eval = Path("data/alpaca_eval.json")

    all_ok = True
    if mt_bench.exists():
        print(f"✓ MT-Bench data found")
    else:
        print(f"✗ MT-Bench data not found. Run: uv run scripts/download_data.py")
        all_ok = False

    if alpaca_eval.exists():
        print(f"✓ AlpacaEval data found")
    else:
        print(f"✗ AlpacaEval data not found. Run: uv run scripts/download_data.py")
        all_ok = False

    return all_ok


def check_hf_auth():
    """Check HuggingFace authentication."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✓ Logged in to HuggingFace as: {user['name']}")
        return True
    except Exception as e:
        print(f"✗ Not logged in to HuggingFace")
        print(f"  Run: uv run huggingface-cli login")
        return False


def main():
    """Run all checks."""
    print("Checking environment setup...\n")

    print("=" * 60)
    print("Dependencies")
    print("=" * 60)
    deps_ok = check_dependencies()

    print("\n" + "=" * 60)
    print("GPU")
    print("=" * 60)
    gpu_ok = check_gpu()

    print("\n" + "=" * 60)
    print("Data")
    print("=" * 60)
    data_ok = check_data()

    print("\n" + "=" * 60)
    print("HuggingFace Authentication")
    print("=" * 60)
    hf_ok = check_hf_auth()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if deps_ok and gpu_ok and data_ok and hf_ok:
        print("✓ All checks passed! Ready to generate completions.")
        print("\nNext steps:")
        print("  1. uv run scripts/generate_completions.py --model-type base")
        print("  2. uv run scripts/generate_completions.py --model-type chat")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
