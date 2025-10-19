#!/usr/bin/env python3
"""Test that models can be loaded and generate a single completion."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_model(model_name: str, model_type: str):
    """Test loading and generating with a model."""
    print(f"\nTesting {model_type} model: {model_name}")
    print("=" * 80)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print("✓ Model loaded")
    print(f"  Device: {model.device}")
    print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Test generation
    print("\nTesting generation...")
    test_prompt = "What is 2+2?"
    if model_type == "chat":
        test_prompt = f"[INST] {test_prompt} [/INST]"

    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    completion = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print(f"\nTest prompt: {test_prompt}")
    print(f"Test completion: {completion[:100]}...")
    print("\n✓ Model test successful!")


def main():
    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument(
        "--model-type",
        choices=["base", "chat"],
        default="base",
        help="Which model to test"
    )
    args = parser.parse_args()

    models = {
        "base": "meta-llama/Llama-2-13b-hf",
        "chat": "meta-llama/Llama-2-13b-chat-hf",
    }

    model_name = models[args.model_type]

    try:
        test_model(model_name, args.model_type)
        print("\n" + "=" * 80)
        print("SUCCESS: Model is ready for full generation!")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
