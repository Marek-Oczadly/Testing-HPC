// Strided global memory access benchmark.
//
// Every launch does the same work: `size` threads each read-modify-write ONE float. Only the
// address pattern changes: thread i touches data[i * stride].
//   stride 1  -> a warp reads 32 consecutive floats (128 B) = 4 fully used 32 B sectors
//   stride 2  -> half of every fetched sector is wasted, ...
//   stride 8+ -> every thread lands in its own 32 B sector, so only 4 of 32 fetched bytes are used
// The useful bytes are constant, so the drop in effective bandwidth is purely the cost of
// uncoalesced access.
//
// Fairness: the same thread count, instruction count and useful bytes for every stride; the
// buffer is allocated once for the largest stride; the working set is larger than the A100's
// 40 MB L2 for every stride (so DRAM, not L2, is measured); warm-up launch before timing.
//
// Build:  nvcc -arch=sm_80 -O3 -std=c++17 -o coalesced-access coalesced-access.cu
// (sm_80 = A100; use sm_90 for H100)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// `size` = number of elements to process (one per thread); data must hold size * stride floats.
__global__ void stride_access(float* data, uint32_t stride, uint32_t size) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // 64-bit index so size * stride cannot overflow
    size_t idx = (size_t)i * stride;
    data[idx] = data[idx] * 1.0001f + 1.0f;
}

int main()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s (sm_%d%d), %d SMs, %.1f GiB, L2 %.1f MiB\n", dev, prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / (1024.0 * 1024 * 1024),
           prop.l2CacheSize / (1024.0 * 1024));

    const uint32_t size = 1u << 25;  // 32M elements = 128 MiB of useful data per direction
    const uint32_t strides[] = {1, 2, 4, 8, 16, 32};
    const uint32_t maxStride = 32;
    const int block = 256;
    const int grid = (size + block - 1) / block;
    const int iters = 20;

    const size_t bufBytes = (size_t)size * maxStride * sizeof(float);
    float* d;
    CUDA_CHECK(cudaMalloc(&d, bufBytes));

    // Read + write of one float per element
    const double usefulBytes = 2.0 * size * sizeof(float);

    printf("stride_access: %u threads, 1 float read + 1 float written each, %d iters\n", size, iters);
    printf("stride | stride (B) | %10s | %12s | %8s | %s\n", "time (ms)", "useful GB/s", "slowdown",
           "check");

    double baseMs = 0.0;
    for (uint32_t stride : strides) {
        CUDA_CHECK(cudaMemset(d, 0, bufBytes));

        // Warm-up (also the launch used for the correctness check: 0 * 1.0001 + 1 == 1)
        stride_access<<<grid, block>>>(d, stride, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Spot-check a few touched elements and one that must be untouched
        bool ok = true;
        const uint32_t probes[] = {0, 1, size / 2, size - 1};
        for (uint32_t p : probes) {
            float v;
            CUDA_CHECK(cudaMemcpy(&v, d + (size_t)p * stride, sizeof(v), cudaMemcpyDeviceToHost));
            ok = ok && v == 1.0f;
        }
        if (stride > 1) {
            float v;
            CUDA_CHECK(cudaMemcpy(&v, d + 1, sizeof(v), cudaMemcpyDeviceToHost));
            ok = ok && v == 0.0f;
        }

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) stride_access<<<grid, block>>>(d, stride, size);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= iters;
        if (stride == 1) baseMs = ms;

        printf("%6u | %10zu | %10.3f | %12.1f | %7.2fx | %s\n", stride, stride * sizeof(float),
               ms, usefulBytes / (ms * 1e6), ms / baseMs, ok ? "PASS" : "FAIL");

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        if (!ok) { cudaFree(d); return EXIT_FAILURE; }
    }

    cudaFree(d);
    return EXIT_SUCCESS;
}
