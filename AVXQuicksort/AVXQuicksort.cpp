#include <bitset>
#include <string>
#include <fstream>
#include <iostream>

#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 

#include "genpermidxs.h"
#include "permidxs8.h"

namespace help {
    // mmprint: print __mm256i register
    void mmprint(__m256i reg)
    {
        std::cout
            << "[ " << _mm256_extract_epi32(reg, 0)
            << ", " << _mm256_extract_epi32(reg, 1)
            << ", " << _mm256_extract_epi32(reg, 2)
            << ", " << _mm256_extract_epi32(reg, 3)
            << ", " << _mm256_extract_epi32(reg, 4)
            << ", " << _mm256_extract_epi32(reg, 5)
            << ", " << _mm256_extract_epi32(reg, 6)
            << ", " << _mm256_extract_epi32(reg, 7)
            << " ]\n";
    }

    // arrprint: print array of size sz
    template<typename T>
    void arrprint(const char* name, T* arr, size_t sz)
    {
        for (int i = 0; i < strlen(name) + 3; i++) std::cout << " ";
        for (int i = 0; i < sz; i++) printf("%2d ", i);
        printf("\n%s = ", name);
        for (int i = 0; i < sz; i++) printf("%2d ", arr[i]);
        std::cout << "\n\n";
    }
};

/// <summary>
/// 
/// </summary>
/// <param name="dst"></param>
/// <param name="src"></param>
/// <param name="sz"></param>
/// <param name="p"></param>
/// <returns></returns>
size_t simple_bipivot_i32x8(int32_t* dst, int32_t* src, size_t sz, int32_t p)
{
    // currently assuming sz % 8 == 0
    __m256i window, cmpres, idxs, shuffled;
    __m256i pivot = _mm256_set1_epi32(p);
    uint16_t mask;
    uint8_t k;
    int32_t* l = dst, * r = dst + sz;

    for (int32_t* sk = src; sk != src + sz; sk += 8)
    {
        window = _mm256_loadu_epi32(sk); // replace w/ stream_load?

        cmpres = _mm256_cmpgt_epi32(pivot, window);
        mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));
        idxs = permidxs8[mask];

        shuffled = _mm256_permutevar8x32_epi32(window, idxs);

        k = __popcnt(mask);
        _mm256_storeu_epi32(l, shuffled); // replace w/ stream
        l += k;

        r -= 8;
        _mm256_storeu_epi32(r, shuffled); // replace w/ stream
        r += k;
    }
    return (l - dst);
}

int main(int argc, char** argv)
{
    const int32_t pivot = 50;
    const size_t sz = 32;
    int32_t src[sz], dst[sz];

    // fill input w/ random numbers
    for (int i = 0; i < sz; i++) src[i] = rand() % 100; 
    help::arrprint("input", src, sz);
    
    // pivot
    size_t i_pivot = simple_bipivot_i32x8(dst, src, sz, pivot);
    help::arrprint("output", dst, sz);
    printf("pivot = %d\tpivot index = %d\n", pivot, i_pivot);
  
    return 0;
}