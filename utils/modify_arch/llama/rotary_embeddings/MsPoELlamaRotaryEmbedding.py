import torch

__all__ = ["MsPoELlamaRotaryEmbedding"]

class MsPoELlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        min_cratio=1,
        max_cratio=3,
        num_heads=32,
        max_position_embeddings=2048,
        base=10000,
        device=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_cratio
        self.max_ratio = max_cratio
        self.num_heads = num_heads

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
        
        # absolute position indices per head
        # t: [num_heads, max_seq_len_cached]
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        ).repeat(num_heads, 1)
        
        # calculate compression ratio per head
        # compress_ratio: [num_heads, 1]
        compress_ratio = torch.arange(
            num_heads, device=device, dtype=self.inv_freq.dtype
        )
        compress_ratio = min_ratio + (max_ratio - min_ratio) * (
            compress_ratio / num_heads
        )
        compress_ratio = compress_ratio.unsqueeze(-1)

        # scale absolute position indices per head with respective compress_ratio
        t = t / compress_ratio
        
        # frequencies: [num_heads, max_seq_len_cached, dim/2]
        # value: t / compress_ratio * inv_freq
        frequencies = torch.einsum("ki,j->kij", t, self.inv_freq)
        
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb: [num_heads, max_seq_len_cached, dim]
        emb = torch.cat((frequencies, frequencies), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # what's the point of x, other than providing dtype and device?
        
        # x: [batch_size, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # for all heads, return cos/sin cache upto seq_len
        # [heads, seq_len, dim] x 2
        return (
            self.cos_cached[:, :seq_len].to(dtype=x.dtype),
            self.sin_cached[:, :seq_len].to(dtype=x.dtype),
        )
