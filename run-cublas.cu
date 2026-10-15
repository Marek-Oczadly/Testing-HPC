// cuBLAS TF32 GEMM benchmark: C = A * B for square sizes N = 128, 1024, 2048, 4096, 8192.
//
// FP32 inputs/outputs are kept in memory; cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 lets
// the Tensor Cores round operands to TF32 (10-bit mantissa) and accumulate in FP32.
// Matrices are row-major and filled identically to run-openblas.cpp (srand(42) per size, same
// order), so the checksums of the two programs can be compared.
//
// Build:  nvcc -arch=sm_90 -O3 -std=c++17 -o run-cublas run-cublas.cu -lcublas
// (sm_90 = H100; use sm_80 for A100, etc.)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st_ = (call);                                              \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__,\
                    (int)st_);                                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Square GEMM of size n x n x n; prints one result line.
static void bench(cublasHandle_t handle, int n)
{
    const size_t elems = (size_t)n * n;

    std::vector<float> hA(elems), hB(elems), hC(elems);
    srand(42);
    for (auto& v : hA) v = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    for (auto& v : hB) v = rand() / (float)RAND_MAX * 2.0f - 1.0f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, elems * sizeof(float)));

    const float alpha = 1.0f, beta = 0.0f;

    // cuBLAS is column-major. A row-major C = A*B is the column-major C^T = B^T * A^T,
    // so pass B first and A second; the buffers are then read/written as row-major C.
    auto gemm = [&]() {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                  &alpha,
                                  dB, CUDA_R_32F, n,
                                  dA, CUDA_R_32F, n,
                                  &beta,
                                  dC, CUDA_R_32F, n,
                                  CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
    };

    // Warm-up (cuBLAS init, heuristics, clocks), then time
    const int warmup = 3;
    for (int i = 0; i < warmup; ++i) gemm();
    CUDA_CHECK(cudaDeviceSynchronize());

    // More iterations for small sizes so the timed region is not dominated by noise
    const double flops = 2.0 * n * (double)n * n;
    const int iters = std::max(5, std::min(1000, (int)(2e13 / flops)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) gemm();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    // Checksum for a coarse cross-check against run-openblas (FP32 vs TF32 will differ slightly)
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (float v : hC) sum += v;

    printf("%5d | %4d iters | %10.3f ms | %10.1f GFLOP/s | checksum %.6f\n", n, iters, ms,
           flops / (ms * 1e6), sum);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s (sm_%d%d), %d SMs, %.1f GiB\n", dev, prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    if (prop.major < 8) {
        fprintf(stderr, "FAIL: TF32 needs compute capability >= 8.0, got %d.%d\n", prop.major, prop.minor);
        return EXIT_FAILURE;
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    printf("cublasGemmEx TF32 (FP32 storage), square N x N x N\n");
    const int sizes[] = {128, 1024, 2048, 4096, 8192};
    for (int n : sizes) bench(handle, n);

    CUBLAS_CHECK(cublasDestroy(handle));
    return EXIT_SUCCESS;
}
