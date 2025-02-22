#pragma once

#include <immintrin.h>
#include <cstdint>
#include <algorithm>

#define LOAD_VECTOR(arr) _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr))
#define STORE_VECTOR(arr, vec)                                                   \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), vec)

const __m256i permutation_masks[256] = { _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0),
                                        _mm256_setr_epi32(0, 2, 3, 4, 5, 6, 7, 1),
                                        _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1),
                                        _mm256_setr_epi32(0, 1, 3, 4, 5, 6, 7, 2),
                                        _mm256_setr_epi32(1, 3, 4, 5, 6, 7, 0, 2),
                                        _mm256_setr_epi32(0, 3, 4, 5, 6, 7, 1, 2),
                                        _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2),
                                        _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 7, 3),
                                        _mm256_setr_epi32(1, 2, 4, 5, 6, 7, 0, 3),
                                        _mm256_setr_epi32(0, 2, 4, 5, 6, 7, 1, 3),
                                        _mm256_setr_epi32(2, 4, 5, 6, 7, 0, 1, 3),
                                        _mm256_setr_epi32(0, 1, 4, 5, 6, 7, 2, 3),
                                        _mm256_setr_epi32(1, 4, 5, 6, 7, 0, 2, 3),
                                        _mm256_setr_epi32(0, 4, 5, 6, 7, 1, 2, 3),
                                        _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3),
                                        _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 7, 4),
                                        _mm256_setr_epi32(1, 2, 3, 5, 6, 7, 0, 4),
                                        _mm256_setr_epi32(0, 2, 3, 5, 6, 7, 1, 4),
                                        _mm256_setr_epi32(2, 3, 5, 6, 7, 0, 1, 4),
                                        _mm256_setr_epi32(0, 1, 3, 5, 6, 7, 2, 4),
                                        _mm256_setr_epi32(1, 3, 5, 6, 7, 0, 2, 4),
                                        _mm256_setr_epi32(0, 3, 5, 6, 7, 1, 2, 4),
                                        _mm256_setr_epi32(3, 5, 6, 7, 0, 1, 2, 4),
                                        _mm256_setr_epi32(0, 1, 2, 5, 6, 7, 3, 4),
                                        _mm256_setr_epi32(1, 2, 5, 6, 7, 0, 3, 4),
                                        _mm256_setr_epi32(0, 2, 5, 6, 7, 1, 3, 4),
                                        _mm256_setr_epi32(2, 5, 6, 7, 0, 1, 3, 4),
                                        _mm256_setr_epi32(0, 1, 5, 6, 7, 2, 3, 4),
                                        _mm256_setr_epi32(1, 5, 6, 7, 0, 2, 3, 4),
                                        _mm256_setr_epi32(0, 5, 6, 7, 1, 2, 3, 4),
                                        _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 6, 7, 5),
                                        _mm256_setr_epi32(1, 2, 3, 4, 6, 7, 0, 5),
                                        _mm256_setr_epi32(0, 2, 3, 4, 6, 7, 1, 5),
                                        _mm256_setr_epi32(2, 3, 4, 6, 7, 0, 1, 5),
                                        _mm256_setr_epi32(0, 1, 3, 4, 6, 7, 2, 5),
                                        _mm256_setr_epi32(1, 3, 4, 6, 7, 0, 2, 5),
                                        _mm256_setr_epi32(0, 3, 4, 6, 7, 1, 2, 5),
                                        _mm256_setr_epi32(3, 4, 6, 7, 0, 1, 2, 5),
                                        _mm256_setr_epi32(0, 1, 2, 4, 6, 7, 3, 5),
                                        _mm256_setr_epi32(1, 2, 4, 6, 7, 0, 3, 5),
                                        _mm256_setr_epi32(0, 2, 4, 6, 7, 1, 3, 5),
                                        _mm256_setr_epi32(2, 4, 6, 7, 0, 1, 3, 5),
                                        _mm256_setr_epi32(0, 1, 4, 6, 7, 2, 3, 5),
                                        _mm256_setr_epi32(1, 4, 6, 7, 0, 2, 3, 5),
                                        _mm256_setr_epi32(0, 4, 6, 7, 1, 2, 3, 5),
                                        _mm256_setr_epi32(4, 6, 7, 0, 1, 2, 3, 5),
                                        _mm256_setr_epi32(0, 1, 2, 3, 6, 7, 4, 5),
                                        _mm256_setr_epi32(1, 2, 3, 6, 7, 0, 4, 5),
                                        _mm256_setr_epi32(0, 2, 3, 6, 7, 1, 4, 5),
                                        _mm256_setr_epi32(2, 3, 6, 7, 0, 1, 4, 5),
                                        _mm256_setr_epi32(0, 1, 3, 6, 7, 2, 4, 5),
                                        _mm256_setr_epi32(1, 3, 6, 7, 0, 2, 4, 5),
                                        _mm256_setr_epi32(0, 3, 6, 7, 1, 2, 4, 5),
                                        _mm256_setr_epi32(3, 6, 7, 0, 1, 2, 4, 5),
                                        _mm256_setr_epi32(0, 1, 2, 6, 7, 3, 4, 5),
                                        _mm256_setr_epi32(1, 2, 6, 7, 0, 3, 4, 5),
                                        _mm256_setr_epi32(0, 2, 6, 7, 1, 3, 4, 5),
                                        _mm256_setr_epi32(2, 6, 7, 0, 1, 3, 4, 5),
                                        _mm256_setr_epi32(0, 1, 6, 7, 2, 3, 4, 5),
                                        _mm256_setr_epi32(1, 6, 7, 0, 2, 3, 4, 5),
                                        _mm256_setr_epi32(0, 6, 7, 1, 2, 3, 4, 5),
                                        _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 7, 6),
                                        _mm256_setr_epi32(1, 2, 3, 4, 5, 7, 0, 6),
                                        _mm256_setr_epi32(0, 2, 3, 4, 5, 7, 1, 6),
                                        _mm256_setr_epi32(2, 3, 4, 5, 7, 0, 1, 6),
                                        _mm256_setr_epi32(0, 1, 3, 4, 5, 7, 2, 6),
                                        _mm256_setr_epi32(1, 3, 4, 5, 7, 0, 2, 6),
                                        _mm256_setr_epi32(0, 3, 4, 5, 7, 1, 2, 6),
                                        _mm256_setr_epi32(3, 4, 5, 7, 0, 1, 2, 6),
                                        _mm256_setr_epi32(0, 1, 2, 4, 5, 7, 3, 6),
                                        _mm256_setr_epi32(1, 2, 4, 5, 7, 0, 3, 6),
                                        _mm256_setr_epi32(0, 2, 4, 5, 7, 1, 3, 6),
                                        _mm256_setr_epi32(2, 4, 5, 7, 0, 1, 3, 6),
                                        _mm256_setr_epi32(0, 1, 4, 5, 7, 2, 3, 6),
                                        _mm256_setr_epi32(1, 4, 5, 7, 0, 2, 3, 6),
                                        _mm256_setr_epi32(0, 4, 5, 7, 1, 2, 3, 6),
                                        _mm256_setr_epi32(4, 5, 7, 0, 1, 2, 3, 6),
                                        _mm256_setr_epi32(0, 1, 2, 3, 5, 7, 4, 6),
                                        _mm256_setr_epi32(1, 2, 3, 5, 7, 0, 4, 6),
                                        _mm256_setr_epi32(0, 2, 3, 5, 7, 1, 4, 6),
                                        _mm256_setr_epi32(2, 3, 5, 7, 0, 1, 4, 6),
                                        _mm256_setr_epi32(0, 1, 3, 5, 7, 2, 4, 6),
                                        _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6),
                                        _mm256_setr_epi32(0, 3, 5, 7, 1, 2, 4, 6),
                                        _mm256_setr_epi32(3, 5, 7, 0, 1, 2, 4, 6),
                                        _mm256_setr_epi32(0, 1, 2, 5, 7, 3, 4, 6),
                                        _mm256_setr_epi32(1, 2, 5, 7, 0, 3, 4, 6),
                                        _mm256_setr_epi32(0, 2, 5, 7, 1, 3, 4, 6),
                                        _mm256_setr_epi32(2, 5, 7, 0, 1, 3, 4, 6),
                                        _mm256_setr_epi32(0, 1, 5, 7, 2, 3, 4, 6),
                                        _mm256_setr_epi32(1, 5, 7, 0, 2, 3, 4, 6),
                                        _mm256_setr_epi32(0, 5, 7, 1, 2, 3, 4, 6),
                                        _mm256_setr_epi32(5, 7, 0, 1, 2, 3, 4, 6),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 7, 5, 6),
                                        _mm256_setr_epi32(1, 2, 3, 4, 7, 0, 5, 6),
                                        _mm256_setr_epi32(0, 2, 3, 4, 7, 1, 5, 6),
                                        _mm256_setr_epi32(2, 3, 4, 7, 0, 1, 5, 6),
                                        _mm256_setr_epi32(0, 1, 3, 4, 7, 2, 5, 6),
                                        _mm256_setr_epi32(1, 3, 4, 7, 0, 2, 5, 6),
                                        _mm256_setr_epi32(0, 3, 4, 7, 1, 2, 5, 6),
                                        _mm256_setr_epi32(3, 4, 7, 0, 1, 2, 5, 6),
                                        _mm256_setr_epi32(0, 1, 2, 4, 7, 3, 5, 6),
                                        _mm256_setr_epi32(1, 2, 4, 7, 0, 3, 5, 6),
                                        _mm256_setr_epi32(0, 2, 4, 7, 1, 3, 5, 6),
                                        _mm256_setr_epi32(2, 4, 7, 0, 1, 3, 5, 6),
                                        _mm256_setr_epi32(0, 1, 4, 7, 2, 3, 5, 6),
                                        _mm256_setr_epi32(1, 4, 7, 0, 2, 3, 5, 6),
                                        _mm256_setr_epi32(0, 4, 7, 1, 2, 3, 5, 6),
                                        _mm256_setr_epi32(4, 7, 0, 1, 2, 3, 5, 6),
                                        _mm256_setr_epi32(0, 1, 2, 3, 7, 4, 5, 6),
                                        _mm256_setr_epi32(1, 2, 3, 7, 0, 4, 5, 6),
                                        _mm256_setr_epi32(0, 2, 3, 7, 1, 4, 5, 6),
                                        _mm256_setr_epi32(2, 3, 7, 0, 1, 4, 5, 6),
                                        _mm256_setr_epi32(0, 1, 3, 7, 2, 4, 5, 6),
                                        _mm256_setr_epi32(1, 3, 7, 0, 2, 4, 5, 6),
                                        _mm256_setr_epi32(0, 3, 7, 1, 2, 4, 5, 6),
                                        _mm256_setr_epi32(3, 7, 0, 1, 2, 4, 5, 6),
                                        _mm256_setr_epi32(0, 1, 2, 7, 3, 4, 5, 6),
                                        _mm256_setr_epi32(1, 2, 7, 0, 3, 4, 5, 6),
                                        _mm256_setr_epi32(0, 2, 7, 1, 3, 4, 5, 6),
                                        _mm256_setr_epi32(2, 7, 0, 1, 3, 4, 5, 6),
                                        _mm256_setr_epi32(0, 1, 7, 2, 3, 4, 5, 6),
                                        _mm256_setr_epi32(1, 7, 0, 2, 3, 4, 5, 6),
                                        _mm256_setr_epi32(0, 7, 1, 2, 3, 4, 5, 6),
                                        _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 0, 7),
                                        _mm256_setr_epi32(0, 2, 3, 4, 5, 6, 1, 7),
                                        _mm256_setr_epi32(2, 3, 4, 5, 6, 0, 1, 7),
                                        _mm256_setr_epi32(0, 1, 3, 4, 5, 6, 2, 7),
                                        _mm256_setr_epi32(1, 3, 4, 5, 6, 0, 2, 7),
                                        _mm256_setr_epi32(0, 3, 4, 5, 6, 1, 2, 7),
                                        _mm256_setr_epi32(3, 4, 5, 6, 0, 1, 2, 7),
                                        _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7),
                                        _mm256_setr_epi32(1, 2, 4, 5, 6, 0, 3, 7),
                                        _mm256_setr_epi32(0, 2, 4, 5, 6, 1, 3, 7),
                                        _mm256_setr_epi32(2, 4, 5, 6, 0, 1, 3, 7),
                                        _mm256_setr_epi32(0, 1, 4, 5, 6, 2, 3, 7),
                                        _mm256_setr_epi32(1, 4, 5, 6, 0, 2, 3, 7),
                                        _mm256_setr_epi32(0, 4, 5, 6, 1, 2, 3, 7),
                                        _mm256_setr_epi32(4, 5, 6, 0, 1, 2, 3, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 4, 7),
                                        _mm256_setr_epi32(1, 2, 3, 5, 6, 0, 4, 7),
                                        _mm256_setr_epi32(0, 2, 3, 5, 6, 1, 4, 7),
                                        _mm256_setr_epi32(2, 3, 5, 6, 0, 1, 4, 7),
                                        _mm256_setr_epi32(0, 1, 3, 5, 6, 2, 4, 7),
                                        _mm256_setr_epi32(1, 3, 5, 6, 0, 2, 4, 7),
                                        _mm256_setr_epi32(0, 3, 5, 6, 1, 2, 4, 7),
                                        _mm256_setr_epi32(3, 5, 6, 0, 1, 2, 4, 7),
                                        _mm256_setr_epi32(0, 1, 2, 5, 6, 3, 4, 7),
                                        _mm256_setr_epi32(1, 2, 5, 6, 0, 3, 4, 7),
                                        _mm256_setr_epi32(0, 2, 5, 6, 1, 3, 4, 7),
                                        _mm256_setr_epi32(2, 5, 6, 0, 1, 3, 4, 7),
                                        _mm256_setr_epi32(0, 1, 5, 6, 2, 3, 4, 7),
                                        _mm256_setr_epi32(1, 5, 6, 0, 2, 3, 4, 7),
                                        _mm256_setr_epi32(0, 5, 6, 1, 2, 3, 4, 7),
                                        _mm256_setr_epi32(5, 6, 0, 1, 2, 3, 4, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 6, 5, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 6, 0, 5, 7),
                                        _mm256_setr_epi32(0, 2, 3, 4, 6, 1, 5, 7),
                                        _mm256_setr_epi32(2, 3, 4, 6, 0, 1, 5, 7),
                                        _mm256_setr_epi32(0, 1, 3, 4, 6, 2, 5, 7),
                                        _mm256_setr_epi32(1, 3, 4, 6, 0, 2, 5, 7),
                                        _mm256_setr_epi32(0, 3, 4, 6, 1, 2, 5, 7),
                                        _mm256_setr_epi32(3, 4, 6, 0, 1, 2, 5, 7),
                                        _mm256_setr_epi32(0, 1, 2, 4, 6, 3, 5, 7),
                                        _mm256_setr_epi32(1, 2, 4, 6, 0, 3, 5, 7),
                                        _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7),
                                        _mm256_setr_epi32(2, 4, 6, 0, 1, 3, 5, 7),
                                        _mm256_setr_epi32(0, 1, 4, 6, 2, 3, 5, 7),
                                        _mm256_setr_epi32(1, 4, 6, 0, 2, 3, 5, 7),
                                        _mm256_setr_epi32(0, 4, 6, 1, 2, 3, 5, 7),
                                        _mm256_setr_epi32(4, 6, 0, 1, 2, 3, 5, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 6, 4, 5, 7),
                                        _mm256_setr_epi32(1, 2, 3, 6, 0, 4, 5, 7),
                                        _mm256_setr_epi32(0, 2, 3, 6, 1, 4, 5, 7),
                                        _mm256_setr_epi32(2, 3, 6, 0, 1, 4, 5, 7),
                                        _mm256_setr_epi32(0, 1, 3, 6, 2, 4, 5, 7),
                                        _mm256_setr_epi32(1, 3, 6, 0, 2, 4, 5, 7),
                                        _mm256_setr_epi32(0, 3, 6, 1, 2, 4, 5, 7),
                                        _mm256_setr_epi32(3, 6, 0, 1, 2, 4, 5, 7),
                                        _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7),
                                        _mm256_setr_epi32(1, 2, 6, 0, 3, 4, 5, 7),
                                        _mm256_setr_epi32(0, 2, 6, 1, 3, 4, 5, 7),
                                        _mm256_setr_epi32(2, 6, 0, 1, 3, 4, 5, 7),
                                        _mm256_setr_epi32(0, 1, 6, 2, 3, 4, 5, 7),
                                        _mm256_setr_epi32(1, 6, 0, 2, 3, 4, 5, 7),
                                        _mm256_setr_epi32(0, 6, 1, 2, 3, 4, 5, 7),
                                        _mm256_setr_epi32(6, 0, 1, 2, 3, 4, 5, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 5, 0, 6, 7),
                                        _mm256_setr_epi32(0, 2, 3, 4, 5, 1, 6, 7),
                                        _mm256_setr_epi32(2, 3, 4, 5, 0, 1, 6, 7),
                                        _mm256_setr_epi32(0, 1, 3, 4, 5, 2, 6, 7),
                                        _mm256_setr_epi32(1, 3, 4, 5, 0, 2, 6, 7),
                                        _mm256_setr_epi32(0, 3, 4, 5, 1, 2, 6, 7),
                                        _mm256_setr_epi32(3, 4, 5, 0, 1, 2, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 4, 5, 3, 6, 7),
                                        _mm256_setr_epi32(1, 2, 4, 5, 0, 3, 6, 7),
                                        _mm256_setr_epi32(0, 2, 4, 5, 1, 3, 6, 7),
                                        _mm256_setr_epi32(2, 4, 5, 0, 1, 3, 6, 7),
                                        _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7),
                                        _mm256_setr_epi32(1, 4, 5, 0, 2, 3, 6, 7),
                                        _mm256_setr_epi32(0, 4, 5, 1, 2, 3, 6, 7),
                                        _mm256_setr_epi32(4, 5, 0, 1, 2, 3, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 5, 4, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 5, 0, 4, 6, 7),
                                        _mm256_setr_epi32(0, 2, 3, 5, 1, 4, 6, 7),
                                        _mm256_setr_epi32(2, 3, 5, 0, 1, 4, 6, 7),
                                        _mm256_setr_epi32(0, 1, 3, 5, 2, 4, 6, 7),
                                        _mm256_setr_epi32(1, 3, 5, 0, 2, 4, 6, 7),
                                        _mm256_setr_epi32(0, 3, 5, 1, 2, 4, 6, 7),
                                        _mm256_setr_epi32(3, 5, 0, 1, 2, 4, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 5, 3, 4, 6, 7),
                                        _mm256_setr_epi32(1, 2, 5, 0, 3, 4, 6, 7),
                                        _mm256_setr_epi32(0, 2, 5, 1, 3, 4, 6, 7),
                                        _mm256_setr_epi32(2, 5, 0, 1, 3, 4, 6, 7),
                                        _mm256_setr_epi32(0, 1, 5, 2, 3, 4, 6, 7),
                                        _mm256_setr_epi32(1, 5, 0, 2, 3, 4, 6, 7),
                                        _mm256_setr_epi32(0, 5, 1, 2, 3, 4, 6, 7),
                                        _mm256_setr_epi32(5, 0, 1, 2, 3, 4, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 0, 5, 6, 7),
                                        _mm256_setr_epi32(0, 2, 3, 4, 1, 5, 6, 7),
                                        _mm256_setr_epi32(2, 3, 4, 0, 1, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 3, 4, 2, 5, 6, 7),
                                        _mm256_setr_epi32(1, 3, 4, 0, 2, 5, 6, 7),
                                        _mm256_setr_epi32(0, 3, 4, 1, 2, 5, 6, 7),
                                        _mm256_setr_epi32(3, 4, 0, 1, 2, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 4, 3, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 4, 0, 3, 5, 6, 7),
                                        _mm256_setr_epi32(0, 2, 4, 1, 3, 5, 6, 7),
                                        _mm256_setr_epi32(2, 4, 0, 1, 3, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 4, 2, 3, 5, 6, 7),
                                        _mm256_setr_epi32(1, 4, 0, 2, 3, 5, 6, 7),
                                        _mm256_setr_epi32(0, 4, 1, 2, 3, 5, 6, 7),
                                        _mm256_setr_epi32(4, 0, 1, 2, 3, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 2, 3, 1, 4, 5, 6, 7),
                                        _mm256_setr_epi32(2, 3, 0, 1, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 3, 2, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 3, 0, 2, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 3, 1, 2, 4, 5, 6, 7),
                                        _mm256_setr_epi32(3, 0, 1, 2, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 0, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 2, 1, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(2, 0, 1, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 0, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7) };



/* partition a single vector, return how many values are greater than pivot,
 * update smallest and largest values in smallest_vec and biggest_vec respectively */
    __forceinline int partition_vec(__m256i& curr_vec, const __m256i& pivot_vec,
        __m256i& smallest_vec, __m256i& biggest_vec) {
    /* which elements are larger than the pivot */
    __m256i compared = _mm256_cmpgt_epi32(curr_vec, pivot_vec);
    /* update the smallest and largest values of the array */
    smallest_vec = _mm256_min_epi32(curr_vec, smallest_vec);
    biggest_vec = _mm256_max_epi32(curr_vec, biggest_vec);
    /* extract the most significant bit from each integer of the vector */
    int mm = _mm256_movemask_ps(_mm256_castsi256_ps(compared));
    /* how many ones, each 1 stands for an element greater than pivot */
    int amount_gt_pivot = _mm_popcnt_u32((mm));
    /* permute elements larger than pivot to the right, and,
     * smaller than or equal to the pivot, to the left */
    curr_vec = _mm256_permutevar8x32_epi32(curr_vec, permutation_masks[mm]);
    /* return how many elements are greater than pivot */
    return amount_gt_pivot;
}

    __forceinline int calc_min(__m256i vec) { /* minimum of 8 int */
        auto perm_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        vec = _mm256_min_epi32(vec, _mm256_permutevar8x32_epi32(vec, perm_mask));
        vec = _mm256_min_epi32(vec, _mm256_shuffle_epi32(vec, 0b10110001));
        vec = _mm256_min_epi32(vec, _mm256_shuffle_epi32(vec, 0b01001110));
        return _mm256_extract_epi32(vec, 0);
    }

    __forceinline int calc_max(__m256i vec) { /* maximum of 8 int */
        auto perm_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        vec = _mm256_max_epi32(vec, _mm256_permutevar8x32_epi32(vec, perm_mask));
        vec = _mm256_max_epi32(vec, _mm256_shuffle_epi32(vec, 0b10110001));
        vec = _mm256_max_epi32(vec, _mm256_shuffle_epi32(vec, 0b01001110));
        return _mm256_extract_epi32(vec, 0);
    }

__forceinline int partition_vectorized_8(int* arr, int left, int right,
    int pivot, int& smallest, int& biggest) {
    /* make array length divisible by eight, shortening the array */
    for (int i = (right - left) % 8; i > 0; --i) {
        smallest = std::min(smallest, arr[left]); biggest = std::max(biggest, arr[left]);
        if (arr[left] > pivot) { std::swap(arr[left], arr[--right]); }
        else { ++left; }
    }

    if (left == right) return left; /* less than 8 elements in the array */

    auto pivot_vec = _mm256_set1_epi32(pivot); /* fill vector with pivot */
    auto sv = _mm256_set1_epi32(smallest); /* vector for smallest elements */
    auto bv = _mm256_set1_epi32(biggest); /* vector for biggest elements */

    if (right - left == 8) { /* if 8 elements left after shortening */
        auto v = LOAD_VECTOR(arr + left);
        int amount_gt_pivot = partition_vec(v, pivot_vec, sv, bv);
        STORE_VECTOR(arr + left, v);
        smallest = calc_min(sv); biggest = calc_max(bv);
        return left + (8 - amount_gt_pivot);
    }

    /* first and last 8 values are partitioned at the end */
    auto vec_left = LOAD_VECTOR(arr + left); /* first 8 values */
    auto vec_right = LOAD_VECTOR(arr + (right - 8)); /* last 8 values  */
    /* store points of the vectors */
    int r_store = right - 8; /* right store point */
    int l_store = left; /* left store point */
    /* indices for loading the elements */
    left += 8; /* increase, because first 8 elements are cached */
    right -= 8; /* decrease, because last 8 elements are cached */

    while (right - left != 0) { /* partition 8 elements per iteration */
        __m256i curr_vec; /* vector to be partitioned */
        /* if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side */
        if ((r_store + 8) - right < left - l_store) {
            right -= 8; curr_vec = LOAD_VECTOR(arr + right);
        }
        else { curr_vec = LOAD_VECTOR(arr + left); left += 8; }
        /* partition the current vector and save it on both sides of the array */
        int amount_gt_pivot = partition_vec(curr_vec, pivot_vec, sv, bv);;
        STORE_VECTOR(arr + l_store, curr_vec); STORE_VECTOR(arr + r_store, curr_vec);
        /* update store points */
        r_store -= amount_gt_pivot; l_store += (8 - amount_gt_pivot);
    }

    /* partition and save vec_left */
    int amount_gt_pivot = partition_vec(vec_left, pivot_vec, sv, bv);
    STORE_VECTOR(arr + l_store, vec_left); STORE_VECTOR(arr + r_store, vec_left);
    l_store += (8 - amount_gt_pivot);
    /* partition and save vec_right */
    amount_gt_pivot = partition_vec(vec_right, pivot_vec, sv, bv);
    STORE_VECTOR(arr + l_store, vec_right);
    l_store += (8 - amount_gt_pivot);

    smallest = calc_min(sv); /* determine smallest value in vector */
    biggest = calc_max(bv); /* determine largest value in vector */
    return l_store;
}

__forceinline int partition_vectorized_64(int* arr, int left, int right,
    int pivot, int& smallest, int& biggest) {
    if (right - left < 129) { /* do not optimize if less than 129 elements */
        return partition_vectorized_8(arr, left, right, pivot, smallest, biggest);
    }

    /* make array length divisible by eight, shortening the array */
    for (int i = (right - left) % 8; i > 0; --i) {
        smallest = std::min(smallest, arr[left]); biggest = std::max(biggest, arr[left]);
        if (arr[left] > pivot) { std::swap(arr[left], arr[--right]); }
        else { ++left; }
    }

    auto pivot_vec = _mm256_set1_epi32(pivot); /* fill vector with pivot */
    auto sv = _mm256_set1_epi32(smallest); /* vector for smallest elements */
    auto bv = _mm256_set1_epi32(biggest); /* vector for biggest elements */

    /* make array length divisible by 64, shortening the array */
    for (int i = ((right - left) % 64) / 8; i > 0; --i) {
        __m256i vec_L = LOAD_VECTOR(arr + left);
        __m256i compared = _mm256_cmpgt_epi32(vec_L, pivot_vec);
        sv = _mm256_min_epi32(vec_L, sv); bv = _mm256_max_epi32(vec_L, bv);
        int mm = _mm256_movemask_ps(_mm256_castsi256_ps(compared));
        int amount_gt_pivot = _mm_popcnt_u32((mm));
        __m256i permuted = _mm256_permutevar8x32_epi32(vec_L, permutation_masks[mm]);

        /* this is a slower way to partition an array with vector instructions */
        __m256i blend_mask = _mm256_cmpgt_epi32(permuted, pivot_vec);
        __m256i vec_R = LOAD_VECTOR(arr + right - 8);
        __m256i vec_L_new = _mm256_blendv_epi8(permuted, vec_R, blend_mask);
        __m256i vec_R_new = _mm256_blendv_epi8(vec_R, permuted, blend_mask);
        STORE_VECTOR(arr + left, vec_L_new); STORE_VECTOR(arr + right - 8, vec_R_new);
        left += (8 - amount_gt_pivot); right -= amount_gt_pivot;
    }

    /* buffer 8 vectors from both sides of the array */
    auto vec_left = LOAD_VECTOR(arr + left), vec_left2 = LOAD_VECTOR(arr + left + 8);
    auto vec_left3 = LOAD_VECTOR(arr + left + 16), vec_left4 = LOAD_VECTOR(arr + left + 24);
    auto vec_left5 = LOAD_VECTOR(arr + left + 32), vec_left6 = LOAD_VECTOR(arr + left + 40);
    auto vec_left7 = LOAD_VECTOR(arr + left + 48), vec_left8 = LOAD_VECTOR(arr + left + 56);
    auto vec_right = LOAD_VECTOR(arr + (right - 64)), vec_right2 = LOAD_VECTOR(arr + (right - 56));
    auto vec_right3 = LOAD_VECTOR(arr + (right - 48)), vec_right4 = LOAD_VECTOR(arr + (right - 40));
    auto vec_right5 = LOAD_VECTOR(arr + (right - 32)), vec_right6 = LOAD_VECTOR(arr + (right - 24));
    auto vec_right7 = LOAD_VECTOR(arr + (right - 16)), vec_right8 = LOAD_VECTOR(arr + (right - 8));

    /* store points of the vectors */
    int r_store = right - 64; /* right store point */
    int l_store = left; /* left store point */
    /* indices for loading the elements */
    left += 64; /* increase because first 64 elements are cached */
    right -= 64; /* decrease because last 64 elements are cached */

    while (right - left != 0) { /* partition 64 elements per iteration */
        __m256i curr_vec, curr_vec2, curr_vec3, curr_vec4, curr_vec5, curr_vec6, curr_vec7, curr_vec8;

        /* if less elements are stored on the right side of the array,
         * then next 8 vectors load from the right side, otherwise load from the left side */
        if ((r_store + 64) - right < left - l_store) {
            right -= 64;
            curr_vec = LOAD_VECTOR(arr + right); curr_vec2 = LOAD_VECTOR(arr + right + 8);
            curr_vec3 = LOAD_VECTOR(arr + right + 16); curr_vec4 = LOAD_VECTOR(arr + right + 24);
            curr_vec5 = LOAD_VECTOR(arr + right + 32); curr_vec6 = LOAD_VECTOR(arr + right + 40);
            curr_vec7 = LOAD_VECTOR(arr + right + 48); curr_vec8 = LOAD_VECTOR(arr + right + 56);
        }
        else {
            curr_vec = LOAD_VECTOR(arr + left); curr_vec2 = LOAD_VECTOR(arr + left + 8);
            curr_vec3 = LOAD_VECTOR(arr + left + 16); curr_vec4 = LOAD_VECTOR(arr + left + 24);
            curr_vec5 = LOAD_VECTOR(arr + left + 32); curr_vec6 = LOAD_VECTOR(arr + left + 40);
            curr_vec7 = LOAD_VECTOR(arr + left + 48); curr_vec8 = LOAD_VECTOR(arr + left + 56);
            left += 64;
        }

        /* partition 8 vectors and store them on both sides of the array */
        int amount_gt_pivot = partition_vec(curr_vec, pivot_vec, sv, bv);
        int amount_gt_pivot2 = partition_vec(curr_vec2, pivot_vec, sv, bv);
        int amount_gt_pivot3 = partition_vec(curr_vec3, pivot_vec, sv, bv);
        int amount_gt_pivot4 = partition_vec(curr_vec4, pivot_vec, sv, bv);
        int amount_gt_pivot5 = partition_vec(curr_vec5, pivot_vec, sv, bv);
        int amount_gt_pivot6 = partition_vec(curr_vec6, pivot_vec, sv, bv);
        int amount_gt_pivot7 = partition_vec(curr_vec7, pivot_vec, sv, bv);
        int amount_gt_pivot8 = partition_vec(curr_vec8, pivot_vec, sv, bv);

        STORE_VECTOR(arr + l_store, curr_vec); l_store += (8 - amount_gt_pivot);
        STORE_VECTOR(arr + l_store, curr_vec2); l_store += (8 - amount_gt_pivot2);
        STORE_VECTOR(arr + l_store, curr_vec3); l_store += (8 - amount_gt_pivot3);
        STORE_VECTOR(arr + l_store, curr_vec4); l_store += (8 - amount_gt_pivot4);
        STORE_VECTOR(arr + l_store, curr_vec5); l_store += (8 - amount_gt_pivot5);
        STORE_VECTOR(arr + l_store, curr_vec6); l_store += (8 - amount_gt_pivot6);
        STORE_VECTOR(arr + l_store, curr_vec7); l_store += (8 - amount_gt_pivot7);
        STORE_VECTOR(arr + l_store, curr_vec8); l_store += (8 - amount_gt_pivot8);

        STORE_VECTOR(arr + r_store + 56, curr_vec); r_store -= amount_gt_pivot;
        STORE_VECTOR(arr + r_store + 56, curr_vec2); r_store -= amount_gt_pivot2;
        STORE_VECTOR(arr + r_store + 56, curr_vec3); r_store -= amount_gt_pivot3;
        STORE_VECTOR(arr + r_store + 56, curr_vec4); r_store -= amount_gt_pivot4;
        STORE_VECTOR(arr + r_store + 56, curr_vec5); r_store -= amount_gt_pivot5;
        STORE_VECTOR(arr + r_store + 56, curr_vec6); r_store -= amount_gt_pivot6;
        STORE_VECTOR(arr + r_store + 56, curr_vec7); r_store -= amount_gt_pivot7;
        STORE_VECTOR(arr + r_store + 56, curr_vec8); r_store -= amount_gt_pivot8;
    }

    /* partition and store 8 vectors coming from the left side of the array */
    int amount_gt_pivot = partition_vec(vec_left, pivot_vec, sv, bv);
    int amount_gt_pivot2 = partition_vec(vec_left2, pivot_vec, sv, bv);
    int amount_gt_pivot3 = partition_vec(vec_left3, pivot_vec, sv, bv);
    int amount_gt_pivot4 = partition_vec(vec_left4, pivot_vec, sv, bv);
    int amount_gt_pivot5 = partition_vec(vec_left5, pivot_vec, sv, bv);
    int amount_gt_pivot6 = partition_vec(vec_left6, pivot_vec, sv, bv);
    int amount_gt_pivot7 = partition_vec(vec_left7, pivot_vec, sv, bv);
    int amount_gt_pivot8 = partition_vec(vec_left8, pivot_vec, sv, bv);

    STORE_VECTOR(arr + l_store, vec_left); l_store += (8 - amount_gt_pivot);
    STORE_VECTOR(arr + l_store, vec_left2); l_store += (8 - amount_gt_pivot2);
    STORE_VECTOR(arr + l_store, vec_left3); l_store += (8 - amount_gt_pivot3);
    STORE_VECTOR(arr + l_store, vec_left4); l_store += (8 - amount_gt_pivot4);
    STORE_VECTOR(arr + l_store, vec_left5); l_store += (8 - amount_gt_pivot5);
    STORE_VECTOR(arr + l_store, vec_left6); l_store += (8 - amount_gt_pivot6);
    STORE_VECTOR(arr + l_store, vec_left7); l_store += (8 - amount_gt_pivot7);
    STORE_VECTOR(arr + l_store, vec_left8); l_store += (8 - amount_gt_pivot8);

    STORE_VECTOR(arr + r_store + 56, vec_left); r_store -= amount_gt_pivot;
    STORE_VECTOR(arr + r_store + 56, vec_left2); r_store -= amount_gt_pivot2;
    STORE_VECTOR(arr + r_store + 56, vec_left3); r_store -= amount_gt_pivot3;
    STORE_VECTOR(arr + r_store + 56, vec_left4); r_store -= amount_gt_pivot4;
    STORE_VECTOR(arr + r_store + 56, vec_left5); r_store -= amount_gt_pivot5;
    STORE_VECTOR(arr + r_store + 56, vec_left6); r_store -= amount_gt_pivot6;
    STORE_VECTOR(arr + r_store + 56, vec_left7); r_store -= amount_gt_pivot7;
    STORE_VECTOR(arr + r_store + 56, vec_left8); r_store -= amount_gt_pivot8;

    /* partition and store 8 vectors coming from the right side of the array */
    amount_gt_pivot = partition_vec(vec_right, pivot_vec, sv, bv);
    amount_gt_pivot2 = partition_vec(vec_right2, pivot_vec, sv, bv);
    amount_gt_pivot3 = partition_vec(vec_right3, pivot_vec, sv, bv);
    amount_gt_pivot4 = partition_vec(vec_right4, pivot_vec, sv, bv);
    amount_gt_pivot5 = partition_vec(vec_right5, pivot_vec, sv, bv);
    amount_gt_pivot6 = partition_vec(vec_right6, pivot_vec, sv, bv);
    amount_gt_pivot7 = partition_vec(vec_right7, pivot_vec, sv, bv);
    amount_gt_pivot8 = partition_vec(vec_right8, pivot_vec, sv, bv);

    STORE_VECTOR(arr + l_store, vec_right); l_store += (8 - amount_gt_pivot);
    STORE_VECTOR(arr + l_store, vec_right2); l_store += (8 - amount_gt_pivot2);
    STORE_VECTOR(arr + l_store, vec_right3); l_store += (8 - amount_gt_pivot3);
    STORE_VECTOR(arr + l_store, vec_right4); l_store += (8 - amount_gt_pivot4);
    STORE_VECTOR(arr + l_store, vec_right5); l_store += (8 - amount_gt_pivot5);
    STORE_VECTOR(arr + l_store, vec_right6); l_store += (8 - amount_gt_pivot6);
    STORE_VECTOR(arr + l_store, vec_right7); l_store += (8 - amount_gt_pivot7);
    STORE_VECTOR(arr + l_store, vec_right8); l_store += (8 - amount_gt_pivot8);

    STORE_VECTOR(arr + r_store + 56, vec_right); r_store -= amount_gt_pivot;
    STORE_VECTOR(arr + r_store + 56, vec_right2); r_store -= amount_gt_pivot2;
    STORE_VECTOR(arr + r_store + 56, vec_right3); r_store -= amount_gt_pivot3;
    STORE_VECTOR(arr + r_store + 56, vec_right4); r_store -= amount_gt_pivot4;
    STORE_VECTOR(arr + r_store + 56, vec_right5); r_store -= amount_gt_pivot5;
    STORE_VECTOR(arr + r_store + 56, vec_right6); r_store -= amount_gt_pivot6;
    STORE_VECTOR(arr + r_store + 56, vec_right7); r_store -= amount_gt_pivot7;
    STORE_VECTOR(arr + r_store + 56, vec_right8);

    smallest = calc_min(sv); biggest = calc_max(bv);
    return l_store;
}


