"""Inference-owned neural-network primitives.

These layers intentionally mirror the corresponding training primitives.  The
inference package owns its copies so that it can be distributed without
depending on the training-only ``modules`` package.  Compatibility is enforced
by strict state-dict loading and logit-equivalence tests.
"""

import torch
import torch.nn as nn


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class RMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-05):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        normalized, dtype = x.float(), x.dtype
        normalized = normalized * torch.rsqrt(
            torch.mean(normalized**2, dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.scale).to(dtype)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear_out = nn.Linear(d_ff // 2, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_out(self.dropout(swiglu(self.linear1(x))))


class TemporalConvPool(nn.Module):
    def __init__(self, dim: int, compression: int = 2, kernel_size: int = 5):
        """Downsample a ``[batch, time, features]`` sequence in time."""
        super().__init__()
        assert compression >= 1, "compression must be >= 1"
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=compression,
            padding=kernel_size // 2,
        )
        self.stride = compression

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)
