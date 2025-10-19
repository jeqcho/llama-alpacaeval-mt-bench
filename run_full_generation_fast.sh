#!/bin/bash
# FAST Full generation workflow for both Llama 2 models
# Uses batch processing for 5-10x speedup!
# Run this in tmux: tmux new -s llama_gen -d "./run_full_generation_fast.sh"

set -e  # Exit on error

export PATH="$HOME/.local/bin:$PATH"

echo "=========================================="
echo "Llama 2 Benchmark Generation (FAST MODE)"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Generate base model completions with batching
echo "=========================================="
echo "Generating Base Model Completions"
echo "Batch size: 8 (processing 8 questions at once)"
echo "=========================================="
uv run scripts/generate_completions_fast.py --model-type base --batch-size 8

echo ""
echo "Base model complete at: $(date)"
echo ""

# Generate chat model completions with batching
echo "=========================================="
echo "Generating Chat Model Completions"
echo "Batch size: 8 (processing 8 questions at once)"
echo "=========================================="
uv run scripts/generate_completions_fast.py --model-type chat --batch-size 8

echo ""
echo "Chat model complete at: $(date)"
echo ""

# Summary
echo "=========================================="
echo "GENERATION COMPLETE!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh outputs/
echo ""
echo "Next step: Upload to HuggingFace"
echo "  uv run scripts/upload_to_hf.py --repo-id YOUR_USERNAME/llama2-benchmarks"
