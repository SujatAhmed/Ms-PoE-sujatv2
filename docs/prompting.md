# utils/lost_in_the_middle/prompting.py

This module builds prompts and structures documents for multi-document QA experiments:

- Defines the `Document` dataclass with metadata fields (title, text, ids, scores, gold flags) and a `from_dict` helper for safe construction.
- Provides `get_qa_prompt`, `get_qa_prompt_index`, and `get_qa_prompt_only_true_index` utilities that format document lists into prompt templates, optionally enforcing answer placement or isolating gold documents.
- Includes `get_closedbook_qa_prompt` and `get_kv_retrieval_prompt` for alternative evaluation formats, sourcing templates from the `prompts` directory.
