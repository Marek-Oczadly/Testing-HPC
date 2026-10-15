// OpenBLAS SGEMM benchmark: C = A * B for square sizes N = 128, 1024, 2048, 4096, 8192, FP32 on the CPU.
//
// CPU baseline for run-cublas.cu. OpenBLAS has no TF32, so this is a full FP32 SGEMM.
// Matrices are row-major and filled identically to run-cublas.cu (srand(42) per size, same
// order), so the checksums of the two programs can be compared.
//
// Build:  g++ -O3 -std=c++17 -o run-openblas run-openblas.cpp -lopenblas
// ILP64 builds (64-bit ints, symbols suffixed with "64_", e.g. libopenblas64_ on some clusters):
//         g++ -O3 -std=c++17 -DBLAS_ILP64 -o run-openblas run-openblas.cpp -lopenblas64_
//         (use -lopenblas64 if that is the library name; check with `ls $(dirname $(which ...))` or the module's lib dir)
// Threads: OPENBLAS_NUM_THREADS=<n> ./run-openblas   (defaults to all cores; match your Slurm allocation)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cblas.h>

// ILP64 OpenBLAS only declares the suffixed names (cblas_sgemm64_, ...)
#ifdef BLAS_ILP64
#define BLAS_FN(name) name##64_
#else
#define BLAS_FN(name) name
#endif

// Square SGEMM of size n x n x n; prints one result line.
static void bench(int n)
{
    const size_t elems = (size_t)n * n;

    std::vector<float> A(elems), B(elems), C(elems, 0.0f);
    srand(42);
    for (auto& v : A) v = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    for (auto& v : B) v = rand() / (float)RAND_MAX * 2.0f - 1.0f;

    const float alpha = 1.0f, beta = 0.0f;

    auto gemm = [&]() {
        BLAS_FN(cblas_sgemm)(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                             alpha, A.data(), n, B.data(), n, beta, C.data(), n);
    };

    // Warm-up (thread pool start-up, page faults on C), then time
    const int warmup = 1;
    for (int i = 0; i < warmup; ++i) gemm();

    // More iterations for small sizes so the timed region is not dominated by noise
    const double flops = 2.0 * n * (double)n * n;
    const int iters = std::max(3, std::min(200, (int)(1e12 / flops)));

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) gemm();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    double sum = 0.0;
    for (float v : C) sum += v;

    printf("%5d | %4d iters | %10.3f ms | %10.1f GFLOP/s | checksum %.6f\n", n, iters, ms,
           flops / (ms * 1e6), sum);
}

int main()
{
    printf("OpenBLAS: %s\n", BLAS_FN(openblas_get_config)());
    printf("Core: %s, threads: %d\n", BLAS_FN(openblas_get_corename)(),
           (int)BLAS_FN(openblas_get_num_threads)());

    printf("cblas_sgemm FP32, square N x N x N\n");
    const int sizes[] = {128, 1024, 2048, 4096, 8192};
    for (int n : sizes) bench(n);

    return EXIT_SUCCESS;
}
