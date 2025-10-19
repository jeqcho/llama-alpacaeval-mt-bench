# Llama 3 8B Generation Quick Start

## Ready to Run!

All Llama 3 scripts are set up. Llama 2 outputs are safe and won't be overwritten.

## Generate Llama 3 Outputs

Run this single command in tmux:

```bash
cd /home/ubuntu/llama-alpacaeval-mt-bench
tmux new -s llama3_gen -d "./run_llama3_generation.sh"
```

This will:
1. Generate base model outputs (meta-llama/Meta-Llama-3-8B)
2. Generate instruct model outputs (meta-llama/Meta-Llama-3-8B-Instruct)
3. Create 6 output files in `outputs/`

## Monitor Progress

```bash
# Reattach to see live progress
tmux attach -t llama3_gen

# Check without attaching
ls -lh outputs/*llama3*

# Check GPU usage
nvidia-smi
```

## Expected Timeline

- **Base model**: ~30-40 minutes (8B model is faster than 13B)
- **Instruct model**: ~30-40 minutes
- **Total**: ~60-80 minutes

## Output Files

Will create in `outputs/`:
- `mt_bench_llama3_base.jsonl` (80 completions)
- `mt_bench_llama3_instruct.jsonl` (80 completions)
- `alpaca_eval_llama3_base.json` (805 completions)
- `alpaca_eval_llama3_instruct.json` (805 completions)
- `llama-3-8b-base.txt` (all outputs concatenated)
- `llama-3-8b-instruct.txt` (all outputs concatenated)

## Upload to HuggingFace

After generation completes:

```bash
uv run scripts/upload_llama3_to_hf.py --username jeqcho
```

This will create 5 new repositories:
- `jeqcho/llama-3-8b-outputs` - All .txt files
- `jeqcho/llama-3-8b-mt-bench` - Base MT-Bench
- `jeqcho/llama-3-8b-alpaca-eval` - Base AlpacaEval
- `jeqcho/llama-3-8b-instruct-mt-bench` - Instruct MT-Bench
- `jeqcho/llama-3-8b-instruct-alpaca-eval` - Instruct AlpacaEval

## Llama 2 Outputs (Already Complete)

Your Llama 2 outputs are safe in `outputs/`:
- ✅ `mt_bench_llama2_base.jsonl`
- ✅ `mt_bench_llama2_chat.jsonl`
- ✅ `alpaca_eval_llama2_base.json`
- ✅ `alpaca_eval_llama2_chat.json`
- ✅ `llama-2-13b-base.txt`
- ✅ `llama-2-13b-chat.txt`

Upload Llama 2 separately if you haven't already:
```bash
uv run scripts/upload_to_hf.py --username jeqcho
```

## Notes

- Llama 3 8B uses different chat template than Llama 2
- Uses proper `<|begin_of_text|>` and `<|start_header_id|>` formatting
- Left padding is configured for optimal batch generation
- No file conflicts - llama3 and llama2 files have different names
