# inference.py

`inference.py` runs the multi-document QA generation pipeline. It loads a pretrained causal language model (with optional Ms-PoE rotary scaling), builds question-answering prompts from the provided dataset, and generates answers in batches. Key responsibilities include:

- Parsing CLI arguments for data paths, model selection, Ms-PoE options, sampling, and batching.
- Seeding numpy and PyTorch RNGs, selecting the CUDA device, and constructing the tokenizer/model via `utils.setup.setup_models`.
- Reading gzipped JSONL QA examples, converting contexts into `Document` objects, and formatting prompts (standard or instruction-style) via `utils.lost_in_the_middle.prompting` helpers.
- Running autoregressive generation with configurable maximum length, then trimming prompts from decoded outputs to collect model responses.
- Writing augmented output records (prompts, documents, model metadata, and answers) back to JSONL for downstream evaluation.
