#!/usr/bin/env python3
"""Generate completions for MT-Bench and AlpacaEval using Llama 2 models."""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_model(model_name: str):
    """Load model and tokenizer with GPU support."""
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token and padding side for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use left padding for decoder-only models
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Model loaded successfully")
    return model, tokenizer


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a single completion."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens (skip the input prompt)
    completion = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return completion


def format_chat_prompt(instruction: str) -> str:
    """Format instruction for Llama 2 Chat model."""
    return f"""[INST] {instruction} [/INST]"""


def generate_mt_bench(model, tokenizer, model_type: str, output_dir: Path, limit: int = None):
    """Generate MT-Bench completions (first turn only)."""
    print("\n" + "="*80)
    print(f"Generating MT-Bench completions with {model_type} model")
    print("="*80)

    # Load questions
    questions_path = Path("data/mt_bench_questions.jsonl")
    questions = []
    with open(questions_path) as f:
        for line in f:
            questions.append(json.loads(line))

    # Limit for testing
    if limit:
        questions = questions[:limit]
        print(f"⚠️  Test mode: Processing only {len(questions)} questions")

    results = []
    all_outputs = []

    for q in tqdm(questions, desc="MT-Bench"):
        question_id = q["question_id"]
        category = q["category"]
        first_turn = q["turns"][0]  # Only use first turn

        # Format prompt based on model type
        if model_type == "chat":
            prompt = format_chat_prompt(first_turn)
        else:
            prompt = first_turn

        # Generate completion
        completion = generate_completion(model, tokenizer, prompt)

        # Store result
        result = {
            "question_id": question_id,
            "category": category,
            "question": first_turn,
            "model_id": model_type,
            "completion": completion,
        }
        results.append(result)
        all_outputs.append(completion)

    # Save JSONL output
    output_path = output_dir / f"mt_bench_llama2_{model_type}.jsonl"
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"\nSaved MT-Bench results to {output_path}")

    return all_outputs


def generate_alpaca_eval(model, tokenizer, model_type: str, output_dir: Path, limit: int = None):
    """Generate AlpacaEval completions."""
    print("\n" + "="*80)
    print(f"Generating AlpacaEval completions with {model_type} model")
    print("="*80)

    # Load questions
    questions_path = Path("data/alpaca_eval.json")
    with open(questions_path) as f:
        questions = json.load(f)

    # Limit for testing
    if limit:
        questions = questions[:limit]
        print(f"⚠️  Test mode: Processing only {len(questions)} questions")

    results = []
    all_outputs = []

    for q in tqdm(questions, desc="AlpacaEval"):
        instruction = q["instruction"]

        # Format prompt based on model type
        if model_type == "chat":
            prompt = format_chat_prompt(instruction)
        else:
            prompt = instruction

        # Generate completion
        completion = generate_completion(model, tokenizer, prompt)

        # Store result
        result = {
            "dataset": q.get("dataset", ""),
            "instruction": instruction,
            "output": completion,
            "generator": f"llama-2-13b-{model_type}",
        }
        results.append(result)
        all_outputs.append(completion)

    # Save JSON output
    output_path = output_dir / f"alpaca_eval_llama2_{model_type}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved AlpacaEval results to {output_path}")

    return all_outputs


def save_concatenated_outputs(mt_outputs: List[str], alpaca_outputs: List[str],
                              model_type: str, output_dir: Path):
    """Save concatenated text outputs."""
    all_outputs = mt_outputs + alpaca_outputs

    output_path = output_dir / f"llama-2-13b-{model_type}.txt"
    with open(output_path, "w") as f:
        for i, output in enumerate(all_outputs):
            f.write(f"--- Output {i+1} ---\n")
            f.write(output)
            f.write("\n\n")

    print(f"\nSaved concatenated outputs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark completions")
    parser.add_argument(
        "--model-type",
        choices=["base", "chat"],
        required=True,
        help="Which Llama 2 model to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions per benchmark (for testing)"
    )
    args = parser.parse_args()

    # Model mapping
    models = {
        "base": "meta-llama/Llama-2-13b-hf",
        "chat": "meta-llama/Llama-2-13b-chat-hf",
    }

    model_name = models[args.model_type]
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load model
    model, tokenizer = load_model(model_name)

    # Generate completions
    mt_outputs = generate_mt_bench(model, tokenizer, args.model_type, output_dir, args.limit)
    alpaca_outputs = generate_alpaca_eval(model, tokenizer, args.model_type, output_dir, args.limit)

    # Save concatenated outputs
    save_concatenated_outputs(mt_outputs, alpaca_outputs, args.model_type, output_dir)

    print("\n" + "="*80)
    print("Generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
