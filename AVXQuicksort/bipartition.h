#pragma once

#include <cstdint>
#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 
#include <tuple>

#include "help.h"
#include "permidxs8.h"


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

// reverse stable. write documentation for this function
__forceinline size_t simple_bipivot_i32x8(int32_t*  dst, int32_t*  src, size_t sz, int32_t p)
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
        uint8_t k = __popcnt(mask);
    
        _mm256_storeu_epi32(l, shuffled);  
        l += k;

        r -= 8;
        _mm256_storeu_epi32(r, shuffled);
        r += k;
    }

    // handle leftover:
    for (int* sk = end; sk != end + rem; sk++)
    {
        if (*sk <= p)   *(l++) = *sk;
        else            *(--r) = *sk;
    }
    return l - dst;
}

__forceinline std::tuple<size_t, size_t, size_t> simple_4pivot_i32x8(int32_t* dst, int32_t* src, size_t sz, int32_t p1, int32_t p2, int32_t p3)
{
    // currently assuming sz % 8 == 0 (fuck!)
    
    size_t i2 = simple_bipivot_i32x8(dst, src, sz, p2);
    size_t i1 = simple_bipivot_i32x8(src, dst, i2, p1);
    size_t i3 = simple_bipivot_i32x8(src+i2, dst+i2, sz-i2, p3);
    
    return std::make_tuple(i1, i2, i3+i2);
}

/// Two pointers 
/// 
size_t twoptr_inplace_bipivot_i32x8(int32_t*  dst, int32_t*  src, size_t sz, int32_t p)
{
    // currently assuming sz % 8 == 0

    __m256i pivot = _mm256_set1_epi32(p);

    int32_t* src_l = src, * src_r = src + sz - 8;
    int32_t* dst_l = dst, * dst_r = dst + sz;
    
    while (src_l < src_r)
    {
        __m256i window_l = _mm256_loadu_epi32(src_l);
        __m256i window_r = _mm256_loadu_epi32(src_r);

        src_l += 8;
        src_r -= 8;

        __m256i cmpres_l = _mm256_cmpgt_epi32(pivot, window_l);
        __m256i cmpres_r = _mm256_cmpgt_epi32(pivot, window_r);

        uint16_t mask_l = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres_l));
        uint16_t mask_r = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres_r));

        __m256i idxs_l = permidxs8[mask_l];
        __m256i idxs_r = permidxs8[mask_r];

        __m256i shuffled_l = _mm256_permutevar8x32_epi32(window_l, idxs_l);
        __m256i shuffled_r = _mm256_permutevar8x32_epi32(window_r, idxs_r);
    
        uint8_t k_l = __popcnt(mask_l);
        uint8_t k_r = __popcnt(mask_r);

        // try rearanging these 
        _mm256_storeu_epi32(dst_l, shuffled_l);  
        dst_l += k_l;
        dst_r -= 8;
        _mm256_storeu_epi32(dst_r, shuffled_l);
        dst_r += k_l;

        _mm256_storeu_epi32(dst_l, shuffled_r);  
        dst_l += k_r;
        dst_r -= 8;
        _mm256_storeu_epi32(dst_r, shuffled_r);
        dst_r += k_r;
    }
    return dst_l - dst;
}

