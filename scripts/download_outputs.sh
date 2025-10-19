#!/bin/bash
# Download all concatenated output .txt files from HuggingFace
# These files contain all 885 completions (80 MT-Bench + 805 AlpacaEval) per model

set -e

echo "=========================================="
echo "Downloading Llama 2 13B Model Outputs"
echo "=========================================="
echo ""

USERNAME="jeqcho"
REPO="llama-2-13b-outputs"
BASE_URL="https://huggingface.co/datasets/${USERNAME}/${REPO}/resolve/main"

# Create outputs directory if it doesn't exist
mkdir -p outputs

echo "Downloading base model outputs..."
wget -O outputs/llama-2-13b-base.txt "${BASE_URL}/llama-2-13b-base.txt"
echo "✓ Downloaded llama-2-13b-base.txt"
echo ""

echo "Downloading chat model outputs..."
wget -O outputs/llama-2-13b-chat.txt "${BASE_URL}/llama-2-13b-chat.txt"
echo "✓ Downloaded llama-2-13b-chat.txt"
echo ""

echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo "Files saved to outputs/"
ls -lh outputs/*.txt
