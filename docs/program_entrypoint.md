# Program entry points

The repository is driven from the shell scripts in `src/`, which invoke `inference.py` to generate answers and `utils/lost_in_the_middle/eval_qa_response.py` to score them.

## Baseline workflow
- Start with `src/baseline.sh`. Each block sets `CUDA_VISIBLE_DEVICES`, runs `python -u inference.py` with a different `--answer_idx` value, and writes predictions under `mdqa_results/`.
- Immediately after each generation call, the script runs `utils/lost_in_the_middle/eval_qa_response.py --input-path <output>` to compute exact-match metrics for that output file.

## Ms-PoE workflow
- Start with `src/ms_poe.sh`. It mirrors the baseline script but adds Ms-PoE flags (`--enable_ms_poe`, `--apply_layers`, `--compress_ratio_min`, `--compress_ratio_max`, and optional `--head_type`).
- As with the baseline script, each invocation of `inference.py` is followed by `eval_qa_response.py` to score the generated answers.

## Under-the-hood entry point
- Both scripts ultimately call `inference.py`, whose `if __name__ == "__main__":` block parses CLI arguments, loads the requested model (standard or Ms-PoE), builds prompts for each QA example, generates answers, and writes augmented JSONL outputs for evaluation.
