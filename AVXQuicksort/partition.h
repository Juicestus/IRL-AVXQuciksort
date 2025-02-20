#pragma once

#include <cstdint>
#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 
#include <tuple>
#include <vector>

#include "help.h"
#include "lookup.h"

//size_t simple_bipivot_i32x8_raw()

// NOTE on loading/storing, we have a few options, namely:
// 
// 1)   _mm256_loadu_epi32(sk); 
//      _mm256_storeu_epi32(l, shuffled);
// 
// 2)   _mm256_loadu_si256(reinterpret_cast<__m256i*>(sk));
//      _mm256_storeu_si256(reinterpret_cast<__m256i*>(l), shuffled);
// 
// 3)   _mm256_stream_load_si256(reinterpret_cast<__m256i*>(sk));
//      _mm256_stream_si256(reinterpret_cast<__m256i*>(l), shuffled);
//
//  When writing outside cache (+ a number of other constraints), #3
//  should be the fastest, however in our case it doesnt help.
//  In addition, b/c out memory is not aligned by 32 byte, it faults.
//  
//  #1 and #2 end up performing the same. #1 is implemented.


template<typename T>    using tuple2 = std::tuple<T, T>;
template<typename T>    using tuple3 = std::tuple<T, T, T>;
template<typename T>    using tuple4 = std::tuple<T, T, T, T>;
template<typename T>    using tuple7 = std::tuple<T, T, T, T, T, T, T>;
template<typename T>    using tuple8 = std::tuple<T, T, T, T, T, T, T, T>;

// Homogenous tuple definition and tie for unwrapping
#define __dectie(...) __VA_ARGS__; std::tie(__VA_ARGS__)


/// Simple bipartition (1 pivot).
/// Input in *src with length sz, result stored in *dst,
/// returns index that marks the partition.
/// 
/// NOTE: could this function be implemented
/// as a composition of *_2dst? -- probably.
/// 
__forceinline size_t legacy_bipartition_i32x8(int32_t* dst, int32_t* src, size_t sz, size_t p)
{
    __m256i pivot = _mm256_set1_epi32(p);
    int32_t* l = dst, * r = dst + sz;
    size_t rem = sz % 8;
    int32_t* end = src + (sz - rem);

    for (int32_t* sk = src; sk != end; sk += 8)
    {
        __m256i window = _mm256_loadu_epi32(sk);

        __m256i cmpres = _mm256_cmpgt_epi32(pivot, window);
        uint16_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));

        __m256i idxs = permidxs8[mask];
        __m256i shuffled = _mm256_permutevar8x32_epi32(window, idxs);
        uint8_t k = popcnts8[mask];     // lookup table for __popcnt(mask), slightly faster.
    
        _mm256_storeu_epi32(l, shuffled);  
        l += k;

        r -= 8;
        _mm256_storeu_epi32(r, shuffled);
        r += k;
    }

    // handle leftover:
    for (int32_t* sk = end; sk != end + rem; sk++)
    {
        if (*sk <= p)   *(l++) = *sk;
        else            *(--r) = *sk;
    }
    return l - dst;
}

/// NOTE: Cleanup pivot vs p etc.

/// Simple bipartition (2-way, 1 pivot), input in *src with length sz.
/// 
/// Pivot must be passed in both size 8 register pivot and int32 p.
/// 
/// Less than partition is written into *dst_l in left to right order.
/// Greated than partition is written into *dst_r in right to left order.
/// 
__forceinline tuple2<size_t> simple_bipartition_2dst_i32x8(int32_t* dst_l, int32_t* dst_r, int32_t* src, size_t sz, __m256i pivot, int32_t p)
{
    int32_t* l = dst_l, * r = dst_l+sz;
    size_t rem = sz % 8;
    int32_t* end = src + (sz - rem);

    for (int32_t* sk = src; sk != end; sk += 8)
    {
        __m256i window = _mm256_loadu_epi32(sk);

        __m256i cmpres = _mm256_cmpgt_epi32(pivot, window);
        uint16_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));

        __m256i idxs = permidxs8[mask];
        __m256i shuffled = _mm256_permutevar8x32_epi32(window, idxs);
        uint8_t k = popcnts8[mask];     // lookup table for __popcnt(mask), slightly faster.

        _mm256_storeu_epi32(l, shuffled);  
        l += k;

        r -= 8;
        _mm256_storeu_epi32(r, shuffled);
        r += k;
    }

    // handle leftover:
    for (int32_t* sk = end; sk != end + rem; sk++)
    {
        if (*sk <= p)   *(l++) = *sk;
        else            *(r--) = *sk;
    }
    
    // how much have the pointers traveled (k_l, k_r):
    return std::make_tuple(l - dst_l, dst_r - r); 
}

/// Simple bipartition (2-way, 1 pivot), input in *src with length sz.
/// Result stored in *dst, partition given by a filled register pivot.
/// Returns index that marks the partition.
/// 
__forceinline size_t simple_bipartition_1dst_i32x8(int32_t* dst, int32_t* src, size_t sz, __m256i pivot, int32_t p)
{
    size_t __dectie(kl, kr) = simple_bipartition_2dst_i32x8(dst, dst + sz, src, sz, pivot, p);
    return kl;      // kr is sexplicitly unused.
}

/// Simple bipartition (2-way, 1 pivot), input in *src with length sz.
/// Result stored in *dst, partition given by pivot.
/// Returns index that marks the partition.
/// 
__forceinline size_t simple_bipartition_1dst_i32x8(int32_t* dst, int32_t* src, size_t sz, int32_t pivot)
{
    return simple_bipartition_1dst_i32x8(dst, src, sz, _mm256_set1_epi32(pivot), pivot);
}


__forceinline tuple3<size_t> simple_4partition_i32x8(int32_t* dst, int32_t* src, size_t sz, tuple3<int32_t> pivs)
{
    size_t p_ctr = simple_bipartition_1dst_i32x8(dst, src, sz, std::get<1>(pivs));
    size_t p_left = simple_bipartition_1dst_i32x8(src, dst, p_ctr, std::get<0>(pivs));
    size_t p_right = simple_bipartition_1dst_i32x8(src+p_ctr, dst+p_ctr, sz-p_ctr, std::get<2>(pivs)) + p_ctr;

    return std::make_tuple(p_left, p_ctr, p_right);
}

__forceinline tuple7<size_t> simple_8partition_i32x8(int32_t* dst, int32_t* src, size_t sz, tuple7<int32_t> pivs)
{
    // pi# = pivot index (size_t)    piv# = pivot (int32_t)

  /*  int32_t piv0, piv1, piv2, piv3, piv4, piv5, piv6;
    std::tie(piv0, piv1, piv2, piv3, piv4, piv5, piv6) = pivs;*/

    int32_t __dectie(piv0, piv1, piv2, piv3, piv4, piv5, piv6) = pivs;

    size_t pi1, pi3, pi5;
    std::tie(pi1, pi3, pi5) = simple_4partition_i32x8(dst, src, sz, std::make_tuple(piv1, piv3, piv5));
    
    size_t pi0 = simple_bipartition_1dst_i32x8(dst, src, pi1, piv0);
    size_t pi2 = simple_bipartition_1dst_i32x8(dst+pi1, src+pi1, pi3-pi1, piv2);
    size_t pi4 = simple_bipartition_1dst_i32x8(dst+pi1, src+pi3, pi5-pi3, piv4);
    size_t pi6 = simple_bipartition_1dst_i32x8(dst+pi1, src+pi3, sz-pi5, piv6);
    
    return std::make_tuple(pi0, pi1, pi2, pi3, pi4, pi5, pi6);
}

/*
size_t twoptr_bipartition_i32x8(int32_t*  dst, int32_t*  src, size_t sz, int32_t p)

*/
