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
#include "partition.h"
//#include "competitor.h"
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

    // ~13.5 b ints/s
    // 
    // 6.9:
    //BENCHMARK(simple_bipartition_1dst_i32x8(dst, src, sz, INT32_MAX / 2));
    //BENCHMARK(simple_bipartition_2dst_i32x8(dst, dst+sz, src, sz, _mm256_set1_epi32(INT32_MAX / 2), INT32_MAX / 2));
    // 
    // 7.4:
    //BENCHMARK(legacy_bipartition_i32x8(dst, src, sz, INT32_MAX / 2));
   
    //  ~6.5 b ints/s
    //BENCHMARK(simple_4partition_i32x8(dst, src, sz, std::make_tuple(INT32_MAX / 4, INT32_MAX / 2, 3 * INT32_MAX / 4)));
   
    //  ~3.3 b ints/s
    /*BENCHMARK(simple_8partition_i32x8(dst, src, sz, std::make_tuple(
        INT32_MAX / 8,          // 1/8 
        INT32_MAX / 4,          // 2/8 = 1/4
        3 * INT32_MAX / 8,      // 3/8
        INT32_MAX / 2,          // 4/8 = 2/4 = 1/2
        5 * INT32_MAX / 8,      // 5/8
        3 * INT32_MAX / 4,      // 6/8 = 3/4
        7 * INT32_MAX / 8       // 7/8 
    )));*/

}
