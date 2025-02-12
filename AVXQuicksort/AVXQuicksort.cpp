#include <iostream>
#include <bitset>

#include <smmintrin.h>
#include <immintrin.h> 

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

void simple_bipivot_i32x8(int32_t* dst, int32_t* src, size_t sz, int32_t p)
{
    // currently assuming sz % 8 == 0
    __m256i window, cmpres;
    __m256i pivot = _mm256_set1_epi32(p);
    uint16_t mask;
    for (int32_t* sk = src; sk != src + sz; sk += 8)
    {
        window = _mm256_loadu_epi32(sk);
        mmprint(window);

        cmpres = _mm256_cmpgt_epi32(window, pivot);
        mmprint(cmpres);
        mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));
        std::cout << std::bitset<8>(mask) << "\n";

        std::cout << "\n------------------------\n";
    }


}

int main(int argc, char** argv)
{
    const size_t sz = 32;
    int32_t src[sz], dst[sz];

    std::cout << "input array = [ ";
    for (int i = 0; i < sz; i++) {
        src[i] = rand() % 100;
        std::cout << src[i] << " ";
    }
    std::cout << " ]\n";

    simple_bipivot_i32x8(dst, src, sz, 50);
}