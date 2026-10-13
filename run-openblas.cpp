// OpenBLAS SGEMM benchmark: C = A * B, 4096 x 4096 x 4096, FP32 on the CPU.
//
// CPU baseline for run-cublas.cu. OpenBLAS has no TF32, so this is a full FP32 SGEMM.
// Matrices are row-major and filled identically to run-cublas.cu (srand(42), same order),
// so the checksums of the two programs can be compared.
//
// Build:  g++ -O3 -std=c++17 -o run-openblas run-openblas.cpp -lopenblas
// Threads: OPENBLAS_NUM_THREADS=<n> ./run-openblas   (defaults to all cores; match your Slurm allocation)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cblas.h>

int main()
{
    printf("OpenBLAS: %s\n", openblas_get_config());
    printf("Core: %s, threads: %d\n", openblas_get_corename(), openblas_get_num_threads());

    const int M = 4096, N = 4096, K = 4096;
    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;

    std::vector<float> A(nA), B(nB), C(nC, 0.0f);
    srand(42);
    for (auto& v : A) v = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    for (auto& v : B) v = rand() / (float)RAND_MAX * 2.0f - 1.0f;

    const float alpha = 1.0f, beta = 0.0f;

    auto gemm = [&]() {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    alpha, A.data(), K, B.data(), N, beta, C.data(), N);
    };

    // Warm-up (thread pool start-up, page faults on C), then time
    const int warmup = 1;
    for (int i = 0; i < warmup; ++i) gemm();

    const int iters = 5;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) gemm();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    printf("cblas_sgemm FP32 %dx%dx%d: %.3f ms, %.1f GFLOP/s\n", M, N, K, ms,
           2.0 * M * N * K / (ms * 1e6));

    double sum = 0.0;
    for (float v : C) sum += v;
    printf("Checksum (sum of C): %.6f\n", sum);

    return EXIT_SUCCESS;
}
