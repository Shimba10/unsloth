# attention_refactor.py

import torch
import inspect
from typing import Optional, Tuple

class UnifiedAttention:
    def __init__(
        self,
        backend: str = "auto",
        causal: bool = True,
        dropout: float = 0.0,
        max_seqlen: Optional[int] = None,
    ):
        """
        Unified interface for multiple attention implementations
        Args:
            backend: One of ['auto', 'xformers', 'flash', 'sdpa', 'flex']
            causal: Apply causal masking
            dropout: Attention dropout probability
            max_seqlen: Maximum sequence length for kernel selection
        """
        self.backend = self._select_backend(backend)
        self.causal = causal
        self.dropout = dropout
        self.max_seqlen = max_seqlen
        self._configure_backend()

    def _select_backend(self, backend: str) -> str:
        """Automatically select best available backend"""
        if backend != "auto":
            return backend
            
        backends = []
        if self._has_xformers():
            backends.append("xformers")
        if self._has_flash_attn():
            backends.append("flash")
        if self._has_flex_attention():
            backends.append("flex")
        if self._has_torch_sdpa():
            backends.append("sdpa")

        return backends[0] if backends else "native"

    def _configure_backend(self):
        """Initialize backend-specific configurations"""
        self._attention_impl = {
            "xformers": self._xformers_attention,
            "flash": self._flash_attention,
            "flex": self._flex_attention,
            "sdpa": self._sdpa_attention,
            "native": self._native_attention,
        }[self.backend]

        # Configure backend-specific optimizations
        if self.backend == "xformers":
            self._enable_xformers_memory_efficient()
        elif self.backend == "flash":
            self._enable_flash_attention()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unified attention forward pass
        Args:
            query: [batch, heads, seq_len, dim]
            key: [batch, heads, seq_len, dim]
            value: [batch, heads, seq_len, dim]
            mask: Optional attention mask
        Returns:
            Attention output tensor
        """
        return self._attention_impl(query, key, value, mask)

    def _xformers_attention(self, q, k, v, mask):
        from xformers.ops import memory_efficient_attention
        return memory_efficient_attention(
            q, k, v, 
            attn_bias=mask,
            p=self.dropout,
            scale=None,
            causal=self.causal,
        )

    def _flash_attention(self, q, k, v, mask):
        from flash_attn import flash_attn_func
        return flash_attn_func(
            q, k, v,
            softmax_scale=None,
            dropout_p=self.dropout,
            causal=self.causal,
            window_size=(-1, -1),
        )

    def _flex_attention(self, q, k, v, mask):
        from flex_attention import flex_attn
        return flex_attn(
            q, k, v,
            mask=mask,
            causal=self.causal,
            dropout=self.dropout,
            max_seqlen=self.max_seqlen,
        )

    def _sdpa_attention(self, q, k, v, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout,
            is_causal=self.causal,
        )

    def _native_attention(self, q, k, v, mask):
        # Fallback native implementation
        scale = q.size(-1) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = attn.softmax(-1)
        return attn @ v

    # Dependency checks
    @staticmethod
    def _has_xformers():
        try:
            import xformers
            return True
        except ImportError:
            return False

    @staticmethod
    def _has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    @staticmethod
    def _has_flex_attention():
        try:
            import flex_attention
            return hasattr(flex_attention, 'flex_attn')
        except ImportError:
            return False

    @staticmethod
    def _has_torch_sdpa():
        return hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    # Optimization configurations
    def _enable_xformers_memory_efficient(self):
        from xformers.ops import xformers_builder
        if not xformers_builder._is_available('memory_efficient_attention'):
            raise RuntimeError("XFormers memory efficient attention unavailable")

    def _enable_flash_attention(self):
        from flash_attn.flash_attention import FlashAttention
        FlashAttention.cuda_kernel_available = lambda: True

# Integration with Unsloth's existing architecture
def patch_attention_layers(model):
    """Replace existing attention layers with unified implementation"""
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            # Create replacement layer with original parameters
            unified_attn = UnifiedAttention(
                causal=module.causal,
                dropout=module.dropout,
                max_seqlen=getattr(module, 'max_seqlen', None),
            )
            
            # Replace forward method
            module.forward = unified_attn.forward
            
            # Preserve original parameters
            module.register_parameter('in_proj_weight', module.in_proj_weight)
            module.register_parameter('in_proj_bias', module.in_proj_bias)
            module.out_proj = module.out_proj

    return model

#Usage

# In Unsloth's model initialization
from attention_refactor import patch_attention_layers

model = AutoModelForCausalLM.from_pretrained(...)
model = patch_attention_layers(model)

# Training proceeds normally with optimized attention