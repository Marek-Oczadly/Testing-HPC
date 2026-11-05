// Vectorised global memory access benchmark.
//
// Every variant is fully coalesced and moves exactly the same data: out[i] = in[i] * a + b over
// n floats. Only the width of each thread's load/store changes:
//   float  ->  LDG.32  / STG.32,  n     threads, 4 B per thread (a warp touches 128 B)
//   float2 ->  LDG.64  / STG.64,  n / 2 threads, 8 B per thread (a warp touches 256 B)
//   float4 ->  LDG.128 / STG.128, n / 4 threads, 16 B per thread (a warp touches 512 B)
// Wider accesses need 2x / 4x fewer load/store instructions, address calculations and threads
// for the same traffic, which relieves the load/store units and lets each SM keep more bytes in
// flight, so a memory-bound kernel gets closer to the DRAM peak.
//
// Fairness: identical useful bytes, identical arithmetic per element, same block size, buffers
// larger than the A100's 40 MB L2 (so DRAM is measured), a warm-up launch before timing, and the
// output of every variant is verified in full. cudaMemcpy device-to-device is timed as a
// reference for the practically achievable bandwidth.
//
// Build:  nvcc -arch=sm_80 -O3 -std=c++17 -o vectorised-access vectorised-access.cu
// (sm_80 = A100; use sm_90 for H100)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// T is float, float2 or float4. `count` = number of T elements (n / floats-per-T); each thread
// handles one T, i.e. one scalar load/store instruction of sizeof(T) bytes.
template <typename T>
__global__ void vector_access(const T* __restrict__ in, T* __restrict__ out, uint32_t count,
                              float a, float b) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    T v = in[i];
    float* f = reinterpret_cast<float*>(&v);
#pragma unroll
    for (int j = 0; j < (int)(sizeof(T) / sizeof(float)); ++j) f[j] = f[j] * a + b;
    out[i] = v;
}

static const float kA = 1.5f, kB = 0.25f;

// Times `launch` (iters back-to-back calls after one warm-up); returns ms per call.
template <typename F>
static float time_ms(F launch, int iters)
{
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

// Runs one variant; returns ms per launch, or a negative value if verification failed.
template <typename T>
static float run(const char* name, const float* dIn, float* dOut, uint32_t n, int block, int iters,
                 double bytes, float baseMs, const std::vector<float>& expected)
{
    const uint32_t width = sizeof(T) / sizeof(float);
    const uint32_t count = n / width;
    const int grid = (count + block - 1) / block;

    auto launch = [&]() {
        vector_access<T><<<grid, block>>>(reinterpret_cast<const T*>(dIn),
                                          reinterpret_cast<T*>(dOut), count, kA, kB);
    };

    CUDA_CHECK(cudaMemset(dOut, 0, (size_t)n * sizeof(float)));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(n);
    CUDA_CHECK(cudaMemcpy(got.data(), dOut, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (uint32_t i = 0; i < n && ok; ++i) ok = got[i] == expected[i];

    float ms = time_ms(launch, iters);
    printf("%-7s | %5u B | %11u | %10.3f | %9.1f | %7.2fx | %s\n", name, width * 4u, count, ms,
           bytes / (ms * 1e6), baseMs > 0 ? baseMs / ms : 1.0f, ok ? "PASS" : "FAIL");
    return ok ? ms : -1.0f;
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

    const uint32_t n = 1u << 26;  // 64M floats = 256 MiB in + 256 MiB out
    const int block = 256;
    const int iters = 50;
    const size_t nbytes = (size_t)n * sizeof(float);
    const double bytes = 2.0 * nbytes;  // read + write

    std::vector<float> hIn(n), expected(n);
    for (uint32_t i = 0; i < n; ++i) {
        hIn[i] = (float)(i % 1024);
        expected[i] = hIn[i] * kA + kB;  // exact in float for these values
    }

    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, nbytes));
    CUDA_CHECK(cudaMalloc(&dOut, nbytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), nbytes, cudaMemcpyHostToDevice));

    printf("vector_access: out = in * a + b over %u floats (%.0f MiB read + %.0f MiB written), "
           "%d iters\n", n, nbytes / 1048576.0, nbytes / 1048576.0, iters);
    printf("%-7s | %7s | %11s | %10s | %9s | %8s | %s\n", "type", "per thr", "threads",
           "time (ms)", "GB/s", "speedup", "check");

    int rc = EXIT_SUCCESS;
    float base = run<float>("float", dIn, dOut, n, block, iters, bytes, 0.0f, expected);
    if (base < 0) rc = EXIT_FAILURE;
    else {
        if (run<float2>("float2", dIn, dOut, n, block, iters, bytes, base, expected) < 0) rc = EXIT_FAILURE;
        if (run<float4>("float4", dIn, dOut, n, block, iters, bytes, base, expected) < 0) rc = EXIT_FAILURE;
    }

    // Reference: driver-tuned device-to-device copy of the same data volume (read + write)
    float cpyMs = time_ms([&]() { cudaMemcpyAsync(dOut, dIn, nbytes, cudaMemcpyDeviceToDevice); },
                          iters);
    printf("%-7s | %7s | %11s | %10.3f | %9.1f | %8s | %s\n", "memcpy", "-", "-", cpyMs,
           bytes / (cpyMs * 1e6), "-", "(reference)");

    cudaFree(dIn); cudaFree(dOut);
    return rc;
}
