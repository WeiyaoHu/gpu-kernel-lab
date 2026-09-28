import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row * n_cols

    # -inf is the neutral padding choice for a max reduction.
    x = tl.load(
        x_ptr + row_start + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    x_max = tl.max(x, axis=0)
    numerator = tl.exp(x - x_max)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(y_ptr + row_start + offsets, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)

    softmax_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


if __name__ == "__main__":
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
    out = softmax(x)
    ref = torch.softmax(x, dim=1)
    print("correct:", torch.allclose(out, ref, atol=1e-5, rtol=1e-5))
