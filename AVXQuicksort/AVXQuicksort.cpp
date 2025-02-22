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

//extern "C" uint64_t power(uint64_t n, uint64_t p);
// [DO NOT CALL RN] extern "C" size_t simple_bipivot_i32x8_ll(int32_t* dst, int32_t* src, size_t sz, int32_t p);

void benchmark_competitor()
{
    // generate dataset and fill with random data
    size_t sz = static_cast<size_t>(1) << 12;
    size_t iters = 1000000;
    int32_t* dst = new int32_t[sz];
    memset(dst, 0, sz);
    int32_t* src = new int32_t[sz];
    std::default_random_engine rng;
    rng.seed(std::random_device{}());
    std::uniform_int_distribution<int32_t> dist(0, INT32_MAX);
    for (int i = 0; i < sz; i++) src[i] = dist(rng);   
    std::cout << "Generated input dataset.\n";

    int32_t p = INT32_MAX / 2;
    int32_t small, big;
    auto t_start = std::chrono::high_resolution_clock::now();                                               
    for (int i = 0; i < iters; i++)
        partition_vectorized_8(src, 0, sz, p, small, big);
    
    auto t_end = std::chrono::high_resolution_clock::now();                                                 
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();            
    std::cout << "Partition completed.\n";
    std::cout << " took " << elapsed_time_ms << "ms to partition " << (iters*sz) << " integers\n";           
    std::cout << " the partition rate = " << (iters*sz) / (elapsed_time_ms * 1000000) << "b integers/s\n";   
}

int main(int argc, char** argv)
{

    //benchmark_competitor();
    //benchmark_bipartition();
     
    benchmark_buckets();


    //test_buckets(64*2);

    return 0;
}
