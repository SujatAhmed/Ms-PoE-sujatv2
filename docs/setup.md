# utils/setup.py

`utils/setup.py` centralizes model/tokenizer loading. Given CLI arguments, it:

- Retrieves model configuration and tokenizer from Hugging Face using optional cache directories.
- Switches between baseline `AutoModelForCausalLM` and the custom `MsPoELlamaForCausalLM` when `--enable_ms_poe` is set.
- Injects Ms-PoE hyperparameters (applied layers, compress ratio range, and head selection strategy) into the configuration before constructing the modified model.
