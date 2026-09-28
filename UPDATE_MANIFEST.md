# Update Manifest

This package contains only the new/supplementary material for the repository after the existing Stage 1–2 files (`00`–`15`).

## CUDA / GEMM additions
- `16_memory_layout_row_major.cu`
- `17_memory_layout_column_major.cu`
- `18_gemm_naive.cu`
- `19_gemm_tiled.cu`
- `20_gemm_register_tiled.cu`
- `21_vectorized_float4.cu`
- `22_gemm_async_pipeline.cu`
- `23_gemm_wmma.cu`
- `24_cublas_sgemm.cu`

## Triton additions
- `25_triton_vector_add.py`
- `26_triton_matrix_add.py`
- `27_triton_row_sum.py`
- `28_triton_softmax.py`
- `29_triton_layernorm.py`
- `30_triton_gemm.py`
- `31_triton_gemm_autotune.py`
- `32_triton_rmsnorm.py`

## Documentation updates
- `README.md`
- `README_CN.md`
- `STAGE3_4_NOTES_CN.md`

The notes consolidate the concepts covered after the current GitHub version: row/column-major layouts, GEMM tiling, register tiling, vectorized access, loop unrolling, double buffering and async copy, Tensor Core/WMMA, cuBLAS, Triton execution model, reductions, Softmax/LayerNorm, GEMM, autotune, and the CUDA/Triton/cuBLAS relationship.
