# src/ms_poe.sh

`ms_poe.sh` mirrors the baseline script but enables Multi-scale Positional Encoding. For each specified answer position, it calls `inference.py` with `--enable_ms_poe` plus layer indices and compression ratios to activate the modified rotary embeddings, then evaluates the generated outputs with `eval_qa_response.py`.
