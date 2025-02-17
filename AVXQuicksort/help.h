#pragma once

#include <bitset>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>

#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 

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

