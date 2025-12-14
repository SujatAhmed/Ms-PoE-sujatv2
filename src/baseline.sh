#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="data/mdqa_10documents.jsonl.gz"
OUTPUT_DIR="mdqa_results"
MODEL_NAME="lmsys/vicuna-7b-v1.5"

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "Input file not found: ${INPUT_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

run_case() {
  local answer_idx="$1"
  local output_file="${OUTPUT_DIR}/baseline-vicuna_7b-10doc-answer${answer_idx}.jsonl"

  CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path "${INPUT_PATH}" \
    --output_path "${output_file}" \
    --model_name "${MODEL_NAME}" \
    --seed 42 \
    --sample_num 500 \
    --answer_idx "${answer_idx}"

  python -u utils/lost_in_the_middle/eval_qa_response.py --input-path "${output_file}"
}

run_case 1
run_case 3
run_case 5
run_case 7
run_case 10


