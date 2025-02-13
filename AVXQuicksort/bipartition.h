#pragma once

#include <cstdint>
#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 
#include "permidxs8.h"

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

size_t simple_bipivot_i32x8(int32_t*  dst, int32_t*  src, size_t sz, int32_t p)
{
    // currently assuming sz % 8 == 0

    __m256i pivot = _mm256_set1_epi32(p);
    int32_t* l = dst, * r = dst + sz;

    for (int32_t* sk = src; sk != src + sz; sk += 8)
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
    return l - dst;
}
