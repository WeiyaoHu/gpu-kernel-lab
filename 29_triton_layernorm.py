import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row * n_cols

    x = tl.load(
        x_ptr + row_start + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    diff = x - mean

    # Mask again here: padded x=0 would otherwise contribute mean^2.
    sq = tl.where(mask, diff * diff, 0.0)
    variance = tl.sum(sq, axis=0) / n_cols
    inv_std = tl.rsqrt(variance + eps)

    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * gamma + beta

    tl.store(y_ptr + row_start + offsets, y, mask=mask)


def layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)

    layernorm_kernel[(n_rows,)](
        x,
        gamma,
        beta,
        out,
        n_cols,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out
