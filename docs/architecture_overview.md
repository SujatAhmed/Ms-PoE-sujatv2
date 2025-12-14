# Project Documentation Overview

This repository implements and evaluates Multi-scale Positional Encoding (Ms-PoE) for long-context language models on multi-document QA benchmarks. The core pieces work together as follows:

1. **Prompt construction** (`utils/lost_in_the_middle/prompting.py`): Takes dataset records and formats them into question-answer prompts with controllable document ordering and indexing. These prompts drive both baseline and Ms-PoE runs.
2. **Model setup** (`utils/setup.py` & `utils/modify_arch/llama.py`): Loads either a standard Hugging Face causal LM or the Ms-PoE-augmented LLaMA variant that introduces head-wise rotary scaling to better retain mid-context information.
3. **Inference pipeline** (`inference.py`): Reads QA examples, builds prompts via the prompting utilities, runs generation with the selected model, and writes enriched outputs that capture prompts, documents, and model answers.
4. **Scoring** (`utils/lost_in_the_middle/eval_qa_response.py` & `utils/lost_in_the_middle/metrics.py`): Evaluates generated answers using normalization and substring-based exact match, producing per-example scores and aggregate metrics.
5. **Experiment orchestration** (`src/baseline.sh` & `src/ms_poe.sh`): Shell scripts that coordinate multiple inference runs across answer positions for both baseline and Ms-PoE models, followed by evaluation.

Together, these components enable reproducible comparison between standard positional encodings and the proposed Ms-PoE approach on long-context QA tasks.

## Where to start
- For baseline runs, execute `src/baseline.sh`; it iterates over several `--answer_idx` values, calls `inference.py` to generate answers, and then scores each file with `eval_qa_response.py`.
- For Ms-PoE runs, execute `src/ms_poe.sh`; it follows the same pattern while enabling the multi-scale positional encoding flags and layer selections.
- Both scripts ultimately enter the `if __name__ == "__main__":` block in `inference.py`, which parses arguments, loads the model, constructs prompts, runs generation, and writes JSONL outputs for downstream scoring.