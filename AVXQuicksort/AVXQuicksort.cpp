#include <bitset>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <random>

#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 

#include "help.h"
#include "generate.h"
#include "competitor.h"
#include "bipartition.h"
#include "bucket.h"
#include "arif_datagen.h"

//extern "C" uint64_t power(uint64_t n, uint64_t p);
// [DO NOT CALL RN] extern "C" size_t simple_bipivot_i32x8_ll(int32_t* dst, int32_t* src, size_t sz, int32_t p);

void benchmark_blacher()
{
    // generate dataset and fill with random data
    size_t sz = static_cast<size_t>(1) << 10;
    size_t iters = 1000000;
    int32_t* dst = new int32_t[sz];
    memset(dst, 0, sz);
    int32_t* src = new int32_t[sz];
    int32_t* base = new int32_t[sz];
#ifdef ARIF_DATAGEN
    datagen::Writer<uint32_t> writer;
    writer.generate((uint32_t*)base, sz, datagen::MT);
#else
    std::default_random_engine rng;
    rng.seed(std::random_device{}());
    std::uniform_int_distribution<int32_t> dist(0, INT32_MAX);
    for (int i = 0; i < sz; i++) base[i] = dist(rng);
#endif
    std::cout << "Generated input dataset.\n";

    int32_t p = INT32_MAX / 2;
#undef small
    int32_t small = INT32_MIN;
    int32_t big = INT32_MAX;
    double elapsed_time_ms = 0;
    for (int i = 0; i < iters; i++) {
        memcpy(src, base, sz);
        auto t_start = std::chrono::high_resolution_clock::now();
        partition_vectorized_8(src, 0, sz, p, small, big);
        auto t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }
   
    std::cout << "Partition completed.\n";
    std::cout << " took " << elapsed_time_ms << "ms to partition " << (iters*sz) << " integers\n";           
    std::cout << " the partition rate = " << (iters*sz) / (elapsed_time_ms * 1000) << " m integers/s\n";   
}

int main(int argc, char** argv)
{
    //benchmark_blacher();
    //benchmark_bipartition();
    benchmark_buckets();


    //test_buckets(64*4);

    return 0;
}
