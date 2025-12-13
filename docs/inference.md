# inference.py

`inference.py` runs the multi-document QA generation pipeline. It loads a pretrained causal language model (with optional Ms-PoE rotary scaling), builds question-answering prompts from the provided dataset, and generates answers in batches. Key responsibilities include:

- Parsing CLI arguments for data paths, model selection, Ms-PoE options, sampling, and batching.
- Seeding numpy and PyTorch RNGs, selecting the CUDA device, and constructing the tokenizer/model via `utils.setup.setup_models`.
- Reading gzipped JSONL QA examples, converting contexts into `Document` objects, and formatting prompts (standard or instruction-style) via `utils.lost_in_the_middle.prompting` helpers.
- Running autoregressive generation with configurable maximum length, then trimming prompts from decoded outputs to collect model responses.
- Writing augmented output records (prompts, documents, model metadata, and answers) back to JSONL for downstream evaluation.

## Functions

- **`set_seed(args)`**: Applies the provided seed to NumPy, PyTorch CPU, and all CUDA devices (when available) so generation results are reproducible across runs.
- **`format_instruct_prompt(instruction)`**: Wraps a QA prompt in an instruction-following template (`### Instruction:`/`### Response:`) used by instruction-tuned models such as LLaMA-Instruct before generation.
- **`chunks(lst, n)`**: Generator that yields successive `n`-sized slices from a list. This powers batch iteration over prompts to keep generation memory usage manageable.

## Main execution flow

The module executes the following steps when run as a script:

1. **Argument parsing**: Defines CLI flags for input/output paths, model selection, Ms-PoE options, prompt filtering (`--only_true`), sampling limits, and batching.
2. **Device and RNG setup**: Selects CUDA when available, records GPU count, and seeds all RNGs via `set_seed`.
3. **Model loading**: Calls `setup_models` to build the tokenizer and model, applies half precision, eval mode, and moves the model to GPU.
4. **Dataset reading and prompt construction**: Streams JSONL examples from `--input_path`, converts contexts to `Document` objects, selects the appropriate prompt builder (true-only or standard), and optionally wraps prompts in the instruction template.
5. **Prompt sampling**: Trims prompts and examples to `--sample_num` if requested and preserves associated documents.
6. **Generation**: Iterates over prompts in batches via `chunks`, tokenizes with optional padding, runs `model.generate`, and resets Ms-PoE layers between batches when enabled.
7. **Post-processing and output**: Decodes each generated sequence, removes the prompt prefix to isolate the model answer, attaches metadata (prompt, documents, model info), and writes one JSON record per example to `--output_path`.