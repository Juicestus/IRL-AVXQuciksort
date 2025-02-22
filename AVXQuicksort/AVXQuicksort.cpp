#include <bitset>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>

#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 

#include "help.h"
#include "generate.h"
//#include "competitor.h"
#include "bipartition.h"
#include "bucket.h"

//extern "C" uint64_t power(uint64_t n, uint64_t p);
// [DO NOT CALL RN] extern "C" size_t simple_bipivot_i32x8_ll(int32_t* dst, int32_t* src, size_t sz, int32_t p);

#define BENCHMARK(FCALL)                                                                                    \
{                                                                                                           \
    const size_t sz = static_cast<size_t>(1) << 12, iters = 1000000;                                        \
    int* src = new int [sz], * dst = new int [sz];                                                          \
    for (int i = 0; i < sz; i++) src[i] = rand() % INT32_MAX;                                               \
    auto t_start = std::chrono::high_resolution_clock::now();                                               \
    for (int i = 0; i < iters; i++) auto ipivs = FCALL;                                                     \
    auto t_end = std::chrono::high_resolution_clock::now();                                                 \
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();            \
    std::cout << "took " << elapsed_time_ms << "ms to partition " << iters * sz << " integers\n";           \
    std::cout << "the partition rate = " << (iters * sz) / (elapsed_time_ms * 1000000) << "b integers/s";   \
}

#define DEMO(FCALL, RESULTS)                            \
{                                                       \
    const size_t sz = 50;                               \
    int* src = new int[sz], * dst = new int[sz];        \
    for (int i = 0; i < sz; i++) src[i] = rand() % 100; \
    help::arrprint("input", src, sz);                   \
    auto pivot = FCALL;                                 \
    RESULTS                                             \
}

int main(int argc, char** argv)
{
    srand(1200);

    // peak of 7.4 (13.5) b ints/s
    //BENCHMARK(bipartition_1_i32x8(dst, src, sz, INT_MAX/2, _mm256_set1_epi32(INT_MAX/2)))
    //BENCHMARK(simple_bipartition_1dst_i32x8(src, sz, INT_MAX / 2, dst))

    //test_buckets(64*2);
     
    // peak of 1.89 b int/s 
    benchmark_buckets();

    return 0;
}
