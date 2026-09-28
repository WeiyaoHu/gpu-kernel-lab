import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
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

    mean_square = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    y = x * inv_rms * weight
    tl.store(y_ptr + row_start + offsets, y, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)

    rmsnorm_kernel[(n_rows,)](
        x,
        weight,
        out,
        n_cols,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out
