#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define SYNAPSE_TARGET(features) __attribute__((target(features)))
#else
#define SYNAPSE_TARGET(features)
#endif

// The kernels vectorize over output columns so each converted f16 weight vector
// is reused across four input rows. Rust handles output-column tails and divides
// row blocks among the configured worker threads.
SYNAPSE_TARGET("avx512f,avx2,f16c,fma")
void synapse_hand_f16_gemm_avx512(size_t m, size_t n, size_t k,
                                  const float *a, const uint16_t *b, float *c) {
    const size_t vector_n = n / 16 * 16;
    size_t row = 0;
    for (; row + 3 < m; row += 4) {
        for (size_t column = 0; column < vector_n; column += 16) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            for (size_t inner = 0; inner < k; ++inner) {
                const __m256i packed = _mm256_loadu_si256(
                    (const __m256i *)(b + inner * n + column));
                const __m512 weight = _mm512_cvtph_ps(packed);
                acc0 = _mm512_fmadd_ps(
                    _mm512_set1_ps(a[(row + 0) * k + inner]), weight, acc0);
                acc1 = _mm512_fmadd_ps(
                    _mm512_set1_ps(a[(row + 1) * k + inner]), weight, acc1);
                acc2 = _mm512_fmadd_ps(
                    _mm512_set1_ps(a[(row + 2) * k + inner]), weight, acc2);
                acc3 = _mm512_fmadd_ps(
                    _mm512_set1_ps(a[(row + 3) * k + inner]), weight, acc3);
            }
            _mm512_storeu_ps(c + (row + 0) * n + column, acc0);
            _mm512_storeu_ps(c + (row + 1) * n + column, acc1);
            _mm512_storeu_ps(c + (row + 2) * n + column, acc2);
            _mm512_storeu_ps(c + (row + 3) * n + column, acc3);
        }
    }
    for (; row < m; ++row) {
        for (size_t column = 0; column < vector_n; column += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (size_t inner = 0; inner < k; ++inner) {
                const __m256i packed = _mm256_loadu_si256(
                    (const __m256i *)(b + inner * n + column));
                const __m512 weight = _mm512_cvtph_ps(packed);
                acc = _mm512_fmadd_ps(
                    _mm512_set1_ps(a[row * k + inner]), weight, acc);
            }
            _mm512_storeu_ps(c + row * n + column, acc);
        }
    }
}

SYNAPSE_TARGET("avx2,f16c,fma")
void synapse_hand_f16_gemm_avx2(size_t m, size_t n, size_t k,
                                const float *a, const uint16_t *b, float *c) {
    const size_t vector_n = n / 8 * 8;
    size_t row = 0;
    for (; row + 3 < m; row += 4) {
        for (size_t column = 0; column < vector_n; column += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            for (size_t inner = 0; inner < k; ++inner) {
                const __m128i packed = _mm_loadu_si128(
                    (const __m128i *)(b + inner * n + column));
                const __m256 weight = _mm256_cvtph_ps(packed);
                acc0 = _mm256_fmadd_ps(
                    _mm256_set1_ps(a[(row + 0) * k + inner]), weight, acc0);
                acc1 = _mm256_fmadd_ps(
                    _mm256_set1_ps(a[(row + 1) * k + inner]), weight, acc1);
                acc2 = _mm256_fmadd_ps(
                    _mm256_set1_ps(a[(row + 2) * k + inner]), weight, acc2);
                acc3 = _mm256_fmadd_ps(
                    _mm256_set1_ps(a[(row + 3) * k + inner]), weight, acc3);
            }
            _mm256_storeu_ps(c + (row + 0) * n + column, acc0);
            _mm256_storeu_ps(c + (row + 1) * n + column, acc1);
            _mm256_storeu_ps(c + (row + 2) * n + column, acc2);
            _mm256_storeu_ps(c + (row + 3) * n + column, acc3);
        }
    }
    for (; row < m; ++row) {
        for (size_t column = 0; column < vector_n; column += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t inner = 0; inner < k; ++inner) {
                const __m128i packed = _mm_loadu_si128(
                    (const __m128i *)(b + inner * n + column));
                const __m256 weight = _mm256_cvtph_ps(packed);
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(a[row * k + inner]), weight, acc);
            }
            _mm256_storeu_ps(c + row * n + column, acc);
        }
    }
}
