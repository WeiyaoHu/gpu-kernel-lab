import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # [BLOCK_M, 1] + [1, BLOCK_N] broadcasts to a 2D data tile.
    offsets = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def matrix_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape and a.ndim == 2

    M, N = a.shape
    out = torch.empty_like(a)
    block_m, block_n = 32, 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    matrix_add_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out
