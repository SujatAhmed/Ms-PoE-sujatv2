import torch

class MyLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        min_compression_ratio=1,
        max_compression_ratio=3,
        num_heads=32,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_prob: float = 1.0,
    ):
        super().__init__()

        # position embedding dimension
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_compression_ratio
        self.max_ratio = max_compression_ratio
        self.num_heads = num_heads
        self.scaling_prob = max(0.0, min(scaling_prob, 1.0))

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        num_heads = self.num_heads
        
        self.max_seq_len_cached = seq_len
        
        relative_position_indices = torch.arange(
            start=-self.max_seq_len_cached, end=self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        ).repeat(num_heads, 1)
        
        compress_ratio = torch.arange(
            num_heads, device=device, dtype=self.inv_freq.dtype
        )
        compress_ratio = min_ratio + (max_ratio - min_ratio) * (
            compress_ratio / num_heads
        )
        compress_ratio = compress_ratio.unsqueeze(-1)

        if self.scaling_prob < 1.0:
            apply_scaling = torch.rand(relative_position_indices.shape, device=device) < self.scaling_prob
            relative_position_indices = torch.where(apply_scaling, relative_position_indices / compress_ratio, relative_position_indices)
        else:
            relative_position_indices = relative_position_indices / compress_ratio
            
        frequencies = torch.einsum("ki,j->kij", relative_position_indices, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :seq_len].to(dtype=x.dtype),
            self.sin_cached[:, :seq_len].to(dtype=x.dtype),
        )
