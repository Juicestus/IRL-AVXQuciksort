#pragma once

#include "partition.h"

#include <cstdint>
#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 
#include <tuple>
#include <vector>

/// Simple bipartition (2-way, 1 pivot), input in *src with length sz.
/// Pivot must be passed into piv (as scalar) and piv_vec (as vector).
/// Less than partition is written into *dst_l in left to right order.
/// Greated than partition is written into *dst_r in right to left order.
__forceinline void bipartition_2dst_i32x8(int32_t*& dst_l, int32_t*& dst_r, int32_t* src, size_t sz, int32_t piv, __m256i piv_vec)
{
    size_t rem = sz % 8;
    int32_t* end = src + (sz - rem);

    for (int32_t* sk = src; sk != end; sk += 8)
    {
        __m256i window = _mm256_loadu_epi32(sk);

        __m256i cmpres = _mm256_cmpgt_epi32(piv_vec, window);
        uint16_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));

        __m256i idxs = permidxs8[mask];
        __m256i shuffled = _mm256_permutevar8x32_epi32(window, idxs);
        uint8_t k = popcnts8[mask];     // lookup table for __popcnt(mask), slightly faster.

        _mm256_storeu_epi32(dst_l, shuffled);
        dst_l += k;

        dst_r -= 8;
        _mm256_storeu_epi32(dst_r, shuffled);
        dst_r += k;
    }

    // handle leftover:
    for (int32_t* sk = end; sk != end + rem; sk++)
    {
        if (*sk <= piv) *(dst_l++) = *sk;
        else            *(dst_r--) = *sk;
    }
}

__forceinline size_t bipartition_1dst_i32x8(int32_t* dst, int32_t* src, size_t sz, int32_t piv, __m256i piv_vec)
{
    int32_t* dst_l = dst, * dst_r = dst + sz;
    bipartition_2dst_i32x8(dst_l, dst_r, src, sz, piv, piv_vec);
    return dst_l - dst;
}

#define CHUNK_SZ 4096   // 2^12

/// Performs 8-way partition on the array beginning at
/// by *src with size sz, around the pivot piv.
/// 
__forceinline void partition_8buckets_i32x8(int32_t* src, size_t sz,
    size_t p0, size_t p1, size_t p2, size_t p3, size_t p4, size_t p5, size_t p6,                                    // Pivots
    int32_t* b0, int32_t* b1, int32_t* b2, int32_t* b3, int32_t* b4, int32_t* b5, int32_t* b6, int32_t* b7,         // Bucket heads
    int32_t*& t0, int32_t*& t1, int32_t*& t2, int32_t*& t3, int32_t*& t4, int32_t*& t5, int32_t*& t6, int32_t*& t7) // bucket Tails
{
    // assume sz % CHUNK_SZ == 0

    int32_t* bf = new int[CHUNK_SZ] {0};    // = to memset?

    __m256i pv0 = _mm256_set1_epi32(p0),    // vectorized pivots
        pv1 = _mm256_set1_epi32(p1),
        pv2 = _mm256_set1_epi32(p2),
        pv3 = _mm256_set1_epi32(p3),
        pv4 = _mm256_set1_epi32(p4),
        pv5 = _mm256_set1_epi32(p5),
        pv6 = _mm256_set1_epi32(p6);
    
    for (int32_t* chunk = src; chunk != src + sz; chunk += CHUNK_SZ)
    {
        // 2 way partition  src --> bf
        size_t k_ctr = bipartition_1dst_i32x8(bf, src, CHUNK_SZ, p3, pv3); 
    
        // 4 way partition  bf --> src
        size_t k_left = bipartition_1dst_i32x8(src, bf, k_ctr, p1, pv1);
        size_t k_right = bipartition_1dst_i32x8(src + k_ctr, bf + k_ctr, CHUNK_SZ-k_ctr, p5, pv5) + k_ctr;
    

    }



}

/*
 const size_t sz = static_cast<size_t>(1) << 12, iters = 1000000;                                        \
    int* src = new int [sz], * dst = new int [sz];                                                          \
    for (int i = 0; i < sz; i++) src[i] = rand() % INT32_MAX;                                               \
    auto t_start = std::chrono::high_resolution_clock::now();                                               \
    for (int i = 0; i < iters; i++) auto ipivs = FCALL;                                                     \
    auto t_end = std::chrono::high_resolution_clock::now();                                                 \
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();            \
    std::cout << "took " << elapsed_time_ms << "ms to partition " << iters * sz << " integers\n";           \
    std::cout << "the partition rate = " << (iters * sz) / (elapsed_time_ms * 1000000) << "b integers/s";   \
*/


__forceinline void benchmark_buckets()
{






}
