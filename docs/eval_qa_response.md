# utils/lost_in_the_middle/eval_qa_response.py

`eval_qa_response.py` scores generated QA answers against gold references:

- Loads model predictions from a JSONL file, preserving example metadata.
- Computes metrics per example (currently best subspan exact match) via `get_metrics_for_example` and aggregates averages across the dataset.
- Optionally writes enriched JSONL output with per-example metric fields for later analysis.
