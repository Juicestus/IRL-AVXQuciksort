#include <bitset>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>

#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 

#include "genpermidxs.h"
#include "permidxs8.h"
#include "bipartition.h"

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


int main(int argc, char** argv)
{
    //src[i] = (int32_t*)(((intptr_t)src[i] + 31) & ~intptr_t(31));
    const int32_t pivot = INT32_MAX/2;

    const size_t sz = static_cast<size_t>(1) << 12;
    const size_t iters = 1000000;

    int32_t* src = new int32_t [sz];
    int32_t* dst = new int32_t [sz];
    for (int i = 0; i < sz; i++) src[i] = rand() % INT32_MAX;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; i++) // 1M times
    {
        size_t i_pivot = simple_bipivot_i32x8(dst, src, sz, pivot);
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "took " << elapsed_time_ms << "ms to partition " << iters * sz << " integers\n";
    std::cout << "the partition rate = " << (iters * sz) / (elapsed_time_ms * 1000000) << "b integers/s";


    //printf("sz = %d\tpivot = %d\tpivot index = %d\n", sz, pivot, i_pivot);
  
    return 0;
}