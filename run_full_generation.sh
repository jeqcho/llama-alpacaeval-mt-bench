#!/bin/bash
# Full generation workflow for both Llama 2 models
# Run this in tmux: tmux new -s llama_gen "./run_full_generation.sh"

set -e  # Exit on error

export PATH="$HOME/.local/bin:$PATH"

echo "=========================================="
echo "Llama 2 Benchmark Generation"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Generate base model completions
echo "=========================================="
echo "Generating Base Model Completions"
echo "=========================================="
uv run scripts/generate_completions.py --model-type base

echo ""
echo "Base model complete at: $(date)"
echo ""

# Generate chat model completions
echo "=========================================="
echo "Generating Chat Model Completions"
echo "=========================================="
uv run scripts/generate_completions.py --model-type chat

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
