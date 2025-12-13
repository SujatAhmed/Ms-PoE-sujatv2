# src/baseline.sh

`baseline.sh` runs the baseline Vicuna-7B evaluation on multi-document QA. It sequences multiple calls to `inference.py` with different answer insertion indices (1, 3, 5, 7, 10), writing outputs to `mdqa_results/`. After each generation pass, it invokes `utils/lost_in_the_middle/eval_qa_response.py` to compute metrics for the produced predictions.
