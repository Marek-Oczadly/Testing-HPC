// Ampere (sm_80+) smoke test.
//
// Ampere features exercised by gemm_bf16_async:
//   * Asynchronous global->shared copies (cp.async) via cuda::pipeline, double buffered
//   * BF16 Tensor Core MMA (WMMA 16x16x16, FP32 accumulate)
// and by count_mismatches:
//   * Warp-level hardware reduction (__reduce_add_sync, sm_80+)
//
// Build:  nvcc -arch=sm_80 -O3 -std=c++17 -o main main.cu
// (use -arch=sm_86 / sm_89 etc. to match the actual GPU)

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/pipeline>
#include <mma.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "This test requires compute capability 8.0 (Ampere) or newer"
#endif

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

constexpr int TILE = 16;   // WMMA tile edge
constexpr int STAGES = 2;  // cp.async pipeline depth

// One warp computes one 16x16 tile of C = A * B (row-major, BF16 in, FP32 out).
// M, N, K must be multiples of 16.
__global__ void gemm_bf16_async(const __nv_bfloat16* __restrict__ A,
                                const __nv_bfloat16* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K)
{
    using namespace nvcuda;

    __shared__ alignas(128) __nv_bfloat16 sA[STAGES][TILE * TILE];
    __shared__ alignas(128) __nv_bfloat16 sB[STAGES][TILE * TILE];

    const int lane = threadIdx.x;  // blockDim.x == 32
    const int tileRow = blockIdx.y * TILE;
    const int tileCol = blockIdx.x * TILE;

    // 32 lanes x 16 bytes = one 16x16 BF16 tile (512 B) per matrix
    const int row = lane / 2;
    const int col = (lane % 2) * 8;

    auto pipe = cuda::make_pipeline();

    auto issue = [&](int stage, int k0) {
        pipe.producer_acquire();
        cuda::memcpy_async(&sA[stage][row * TILE + col],
                           A + (size_t)(tileRow + row) * K + k0 + col,
                           cuda::aligned_size_t<16>(16), pipe);
        cuda::memcpy_async(&sB[stage][row * TILE + col],
                           B + (size_t)(k0 + row) * N + tileCol + col,
                           cuda::aligned_size_t<16>(16), pipe);
        pipe.producer_commit();
    };

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int kTiles = K / TILE;
    issue(0, 0);

    for (int kt = 0; kt < kTiles; ++kt) {
        const int cur = kt % STAGES;
        if (kt + 1 < kTiles) {
            issue((kt + 1) % STAGES, (kt + 1) * TILE);
            cuda::pipeline_consumer_wait_prior<1>(pipe);  // current stage landed, next in flight
        } else {
            cuda::pipeline_consumer_wait_prior<0>(pipe);
        }
        __syncwarp();  // copies were issued by other lanes

        wmma::load_matrix_sync(fa, sA[cur], TILE);
        wmma::load_matrix_sync(fb, sB[cur], TILE);
        wmma::mma_sync(acc, fa, fb, acc);

        __syncwarp();  // all lanes done reading before this buffer is refilled
        pipe.consumer_release();
    }

    wmma::store_matrix_sync(C + (size_t)tileRow * N + tileCol, acc, N, wmma::mem_row_major);
}

// Plain FP32 reference on the same BF16-rounded inputs.
__global__ void gemm_reference(const __nv_bfloat16* A, const __nv_bfloat16* B,
                               float* C, int M, int N, int K)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= M || c >= N) return;
    float s = 0.0f;
    for (int k = 0; k < K; ++k)
        s += __bfloat162float(A[(size_t)r * K + k]) * __bfloat162float(B[(size_t)k * N + c]);
    C[(size_t)r * N + c] = s;
}

// Counts out-of-tolerance elements; per-warp sum via __reduce_add_sync (sm_80+).
__global__ void count_mismatches(const float* test, const float* ref, size_t n,
                                 unsigned* bad)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mismatch = 0;
    if (i < n) {
        float diff = fabsf(test[i] - ref[i]);
        mismatch = (diff > 1e-3f + 1e-3f * fabsf(ref[i])) ? 1u : 0u;
    }
    unsigned warpTotal = __reduce_add_sync(0xffffffffu, mismatch);
    if ((threadIdx.x & 31) == 0 && warpTotal) atomicAdd(bad, warpTotal);
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
        fprintf(stderr, "FAIL: need compute capability >= 8.0, got %d.%d\n", prop.major, prop.minor);
        return EXIT_FAILURE;
    }

    const int M = 1024, N = 1024, K = 1024;
    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;

    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    srand(42);
    for (auto& v : hA) v = __float2bfloat16(rand() / (float)RAND_MAX * 2.0f - 1.0f);
    for (auto& v : hB) v = __float2bfloat16(rand() / (float)RAND_MAX * 2.0f - 1.0f);

    __nv_bfloat16 *dA, *dB;
    float *dC, *dRef;
    unsigned* dBad;
    CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(*dA)));
    CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(*dB)));
    CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef, nC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBad, sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(*dA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(*dB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dBad, 0, sizeof(unsigned)));

    dim3 grid(N / TILE, M / TILE), block(32);

    // Warm-up, then time
    gemm_bf16_async<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 20;
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) gemm_bf16_async<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;
    printf("gemm_bf16_async %dx%dx%d: %.3f ms, %.1f GFLOP/s\n", M, N, K, ms,
           2.0 * M * N * K / (ms * 1e6));

    // Verify
    dim3 rblock(16, 16), rgrid((N + 15) / 16, (M + 15) / 16);
    gemm_reference<<<rgrid, rblock>>>(dA, dB, dRef, M, N, K);
    count_mismatches<<<(unsigned)((nC + 255) / 256), 256>>>(dC, dRef, nC, dBad);
    CUDA_CHECK(cudaGetLastError());
    unsigned bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, dBad, sizeof(bad), cudaMemcpyDeviceToHost));

    printf("Mismatches vs reference: %u / %zu\n", bad, nC);
    printf(bad == 0 ? "PASS\n" : "FAIL\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef); cudaFree(dBad);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return bad == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
