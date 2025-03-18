import torch
from torch.nn import functional as F
from typing import Optional, Tuple

class FlexAttention(torch.nn.Module):
    def __init__(
        self,
        window_size: Optional[int] = None,
        causal: bool = True,
        use_packed: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        self.use_packed = use_packed
        self._compiled_kernel = None

    def _get_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        packed_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.use_packed and packed_mask is not None:
            mask = packed_mask.unsqueeze(1).expand(-1, seq_len, -1)
        elif self.window_size:
            mask = self._sliding_window_mask(seq_len, device, dtype)
        elif self.causal:
            mask = self._causal_mask(seq_len, device, dtype)
        else:
            mask = None
        return mask

    def _causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), 1)

    def _sliding_window_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = 1
        return mask

    def _compile_kernel(self, q: torch.Tensor, k: torch.Tensor):
        if self._compiled_kernel is None or self._compiled_kernel.dtype != q.dtype:
            from triton import cdiv, jit
            
            @jit
            def flex_attn_kernel(
                Q, K, V, M, Out,
                stride_qz, stride_qh, stride_qm, stride_qk,
                stride_kz, stride_kh, stride_kn, stride_kk,
                seq_len, head_dim,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,):
                # Simplified Triton kernel implementation
                # ... (actual kernel code would go here)

                self._compiled_kernel = flex_attn_kernel
        return self._compiled_kernel

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        packed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Generate appropriate attention mask
        attn_mask = self._get_attention_mask(
            seq_len, query.device, query.dtype, packed_mask
        )

        # Use PyTorch's optimized SDPA if possible
        if not self.use_packed and self.window_size is None:
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, is_causal=self.causal
            )

        # Fallback to custom Triton kernel for special cases
        self._compile_kernel(query, key)
        output = torch.empty_like(query)
        
        # Call compiled kernel with dynamic grid size
        grid = (cdiv(seq_len, BLOCK_M), num_heads, batch_size)
        self._compiled_kernel[grid](
            query, key, value, attn_mask, output,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            seq_len, head_dim,
            BLOCK_M=128, BLOCK_N=64,
        )
        return output

    def reset_parameters(self):
        self._compiled_kernel = None
        
        
#Replace Existing Attention Layers

def replace_attention_layers(model):
    for module in model.modules():
        if isinstance(module, Attention):
            flex_attn = FlexAttention(
                window_size=module.window_size,
                causal=module.causal,
                use_packed=module.use_packed,
            )
            module.forward = flex_attn.forward
    return model


#Training Configuration

model = AutoModel.from_pretrained(...)
model = replace_attention_layers(model)

# Mixed sequence lengths in same batch
trainer = Trainer(
    model=model,
    data_collator=DynamicPadCollator(),
    # ... other args
)