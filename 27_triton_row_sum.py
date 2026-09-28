import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(
        x_ptr + row * n_cols + offsets,
        mask=mask,
        other=0.0,  # neutral element for sum
    )
    row_sum = tl.sum(x, axis=0)
    tl.store(out_ptr + row, row_sum)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    block_size = triton.next_power_of_2(n_cols)

    row_sum_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return out
