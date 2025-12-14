# utils/modify_arch/llama.py

`utils/modify_arch/llama.py` defines the Ms-PoE extensions to the LLaMA architecture. Highlights:

- Implements `MsPoELlamaRotaryEmbedding`, which builds head-specific rotary caches scaled by a range of compression ratios to fuse multiple positional scales.
- Adds `MsPoELlamaAttention` that replaces standard attention with head-wise rotary scaling, optional head ordering based on outlier statistics, and caching utilities to reset rotary states after generation.
- Wraps the full language model via `MsPoELlamaForCausalLM`, overriding generation setup to plug in the multi-scale rotary embeddings while reusing Hugging Face components for projection layers, causal masks, and key/value repetition.
