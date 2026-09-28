import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # A Triton program is a computation instance. BLOCK_SIZE describes the
    # logical data tile handled by that program; it is not a CUDA thread count.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape

    out = torch.empty_like(x)
    n_elements = out.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)

    vector_add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return out


if __name__ == "__main__":
    x = torch.randn(1_000_000, device="cuda")
    y = torch.randn_like(x)
    ref = x + y
    out = vector_add(x, y)
    print("correct:", torch.allclose(out, ref, atol=1e-6, rtol=1e-6))
