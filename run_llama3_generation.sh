#!/bin/bash
# FAST Full generation workflow for Llama 3 8B models (base and instruct)
# Uses batch processing for 5-10x speedup!
# Run this in tmux: tmux new -s llama3_gen -d "./run_llama3_generation.sh"

set -e  # Exit on error

export PATH="$HOME/.local/bin:$PATH"

echo "=========================================="
echo "Llama 3.1 8B Benchmark Generation (FAST MODE)"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Generate base model completions with batching
echo "=========================================="
echo "Generating Base Model Completions"
echo "Model: meta-llama/Llama-3.1-8B"
echo "Batch size: 8 (processing 8 questions at once)"
echo "=========================================="
uv run scripts/generate_llama3.py --model-type base --batch-size 8

echo ""
echo "Base model complete at: $(date)"
echo ""

# Generate instruct model completions with batching
echo "=========================================="
echo "Generating Instruct Model Completions"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "Batch size: 8 (processing 8 questions at once)"
echo "=========================================="
uv run scripts/generate_llama3.py --model-type instruct --batch-size 8

echo ""
echo "Instruct model complete at: $(date)"
echo ""

# Summary
echo "=========================================="
echo "GENERATION COMPLETE!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh outputs/*llama3*
echo ""
echo "Next step: Upload to HuggingFace"
echo "  uv run scripts/upload_llama3_to_hf.py --username jeqcho"
