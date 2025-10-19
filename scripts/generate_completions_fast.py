#!/usr/bin/env python3
"""Generate completions for MT-Bench and AlpacaEval using Llama 2 models - OPTIMIZED VERSION."""

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

    # Use left padding for batch generation with decoder-only models
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Model loaded successfully")
    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Generate completions for a batch of prompts."""
    # Tokenize all prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
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

    # Decode completions
    completions = []
    for i, output in enumerate(outputs):
        # Get the length of the input to skip it
        input_length = inputs["input_ids"][i].shape[0]
        completion = tokenizer.decode(
            output[input_length:],
            skip_special_tokens=True
        )
        completions.append(completion)

    return completions


def format_chat_prompt(instruction: str) -> str:
    """Format instruction for Llama 2 Chat model."""
    return f"""[INST] {instruction} [/INST]"""


def generate_mt_bench(model, tokenizer, model_type: str, output_dir: Path, limit: int = None, batch_size: int = 8):
    """Generate MT-Bench completions (first turn only)."""
    print("\n" + "="*80)
    print(f"Generating MT-Bench completions with {model_type} model (batch_size={batch_size})")
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

    # Process in batches
    for i in tqdm(range(0, len(questions), batch_size), desc="MT-Bench"):
        batch = questions[i:i + batch_size]

        # Prepare prompts
        prompts = []
        for q in batch:
            first_turn = q["turns"][0]
            if model_type == "chat":
                prompt = format_chat_prompt(first_turn)
            else:
                prompt = first_turn
            prompts.append(prompt)

        # Generate batch
        completions = generate_batch(model, tokenizer, prompts)

        # Store results
        for q, completion in zip(batch, completions):
            result = {
                "question_id": q["question_id"],
                "category": q["category"],
                "question": q["turns"][0],
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


def generate_alpaca_eval(model, tokenizer, model_type: str, output_dir: Path, limit: int = None, batch_size: int = 8):
    """Generate AlpacaEval completions."""
    print("\n" + "="*80)
    print(f"Generating AlpacaEval completions with {model_type} model (batch_size={batch_size})")
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

    # Process in batches
    for i in tqdm(range(0, len(questions), batch_size), desc="AlpacaEval"):
        batch = questions[i:i + batch_size]

        # Prepare prompts
        prompts = []
        for q in batch:
            instruction = q["instruction"]
            if model_type == "chat":
                prompt = format_chat_prompt(instruction)
            else:
                prompt = instruction
            prompts.append(prompt)

        # Generate batch
        completions = generate_batch(model, tokenizer, prompts)

        # Store results
        for q, completion in zip(batch, completions):
            result = {
                "dataset": q.get("dataset", ""),
                "instruction": q["instruction"],
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
    parser = argparse.ArgumentParser(description="Generate benchmark completions (FAST version)")
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8, larger = faster but more VRAM)"
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
    mt_outputs = generate_mt_bench(model, tokenizer, args.model_type, output_dir, args.limit, args.batch_size)
    alpaca_outputs = generate_alpaca_eval(model, tokenizer, args.model_type, output_dir, args.limit, args.batch_size)

    # Save concatenated outputs
    save_concatenated_outputs(mt_outputs, alpaca_outputs, args.model_type, output_dir)

    print("\n" + "="*80)
    print("Generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
