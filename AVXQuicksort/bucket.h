#pragma once

#include <cstdint>
#include <random>
#include <iostream>
#include <chrono>

#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 

#include "arif_datagen.h"
#include "lookup.h"

__forceinline void chunk_8x8(int32_t*& dst_l, int32_t*& dst_r, int32_t*& sk, int32_t* end, int32_t piv, __m256i piv_vec)
{
    for ( ; sk != end; sk += 64)
    {
        __m256i window0 = _mm256_loadu_epi32(sk);                           // hold ymm 0..7
        __m256i window1 = _mm256_loadu_epi32(sk+8);
        __m256i window2 = _mm256_loadu_epi32(sk+16);
        __m256i window3 = _mm256_loadu_epi32(sk+24);
        __m256i window4 = _mm256_loadu_epi32(sk+32);
        __m256i window5 = _mm256_loadu_epi32(sk+40);
        __m256i window6 = _mm256_loadu_epi32(sk+48);
        __m256i window7 = _mm256_loadu_epi32(sk+56);

        __m256i cmpres0 = _mm256_cmpgt_epi32(piv_vec, window0);             // hold ymm 8..15
        __m256i cmpres1 = _mm256_cmpgt_epi32(piv_vec, window1);
        __m256i cmpres2 = _mm256_cmpgt_epi32(piv_vec, window2);
        __m256i cmpres3 = _mm256_cmpgt_epi32(piv_vec, window3);
        __m256i cmpres4 = _mm256_cmpgt_epi32(piv_vec, window4);
        __m256i cmpres5 = _mm256_cmpgt_epi32(piv_vec, window5);
        __m256i cmpres6 = _mm256_cmpgt_epi32(piv_vec, window6);
        __m256i cmpres7 = _mm256_cmpgt_epi32(piv_vec, window7);

        uint16_t mask0 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres0));  // free ymm 8..15
        uint16_t mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres1));  
        uint16_t mask2 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres2));
        uint16_t mask3 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres3));
        uint16_t mask4 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres4));
        uint16_t mask5 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres5));
        uint16_t mask6 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres6));
        uint16_t mask7 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres7));

        __m256i idxs0 = permidxs8[mask0];                                   // hold ymm 8..15
        __m256i idxs1 = permidxs8[mask1];
        __m256i idxs2 = permidxs8[mask2];
        __m256i idxs3 = permidxs8[mask3];
        __m256i idxs4 = permidxs8[mask4];
        __m256i idxs5 = permidxs8[mask5];
        __m256i idxs6 = permidxs8[mask6];
        __m256i idxs7 = permidxs8[mask7];

        __m256i shuffled0 = _mm256_permutevar8x32_epi32(window0, idxs0);    // free ymm 0..7
        __m256i shuffled1 = _mm256_permutevar8x32_epi32(window1, idxs1);    // free ymm 8..15
        __m256i shuffled2 = _mm256_permutevar8x32_epi32(window2, idxs2);    // hold ymm 0..7
        __m256i shuffled3 = _mm256_permutevar8x32_epi32(window3, idxs3);
        __m256i shuffled4 = _mm256_permutevar8x32_epi32(window4, idxs4);
        __m256i shuffled5 = _mm256_permutevar8x32_epi32(window5, idxs5);
        __m256i shuffled6 = _mm256_permutevar8x32_epi32(window6, idxs6);
        __m256i shuffled7 = _mm256_permutevar8x32_epi32(window7, idxs7);

        uint8_t k0 = popcnts8[mask0];     // lookup table for __popcnt(mask), slightly faster.
        uint8_t k1 = popcnts8[mask1];     
        uint8_t k2 = popcnts8[mask2];     
        uint8_t k3 = popcnts8[mask3];     
        uint8_t k4 = popcnts8[mask4];     
        uint8_t k5 = popcnts8[mask5];     
        uint8_t k6 = popcnts8[mask6];     
        uint8_t k7 = popcnts8[mask7];     
    
        _mm256_storeu_epi32(dst_l, shuffled0);   dst_l += k0;               // free ymm 0..7
        _mm256_storeu_epi32(dst_l, shuffled1);   dst_l += k1;
        _mm256_storeu_epi32(dst_l, shuffled2);   dst_l += k2;
        _mm256_storeu_epi32(dst_l, shuffled3);   dst_l += k3;
        _mm256_storeu_epi32(dst_l, shuffled4);   dst_l += k4;
        _mm256_storeu_epi32(dst_l, shuffled5);   dst_l += k5;
        _mm256_storeu_epi32(dst_l, shuffled6);   dst_l += k6;
        _mm256_storeu_epi32(dst_l, shuffled7);   dst_l += k7;

        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled0); dst_r += k0;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled1); dst_r += k1;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled2); dst_r += k2;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled3); dst_r += k3;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled4); dst_r += k4;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled5); dst_r += k5;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled6); dst_r += k6;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled7); dst_r += k7;
        // could replace above w/ dst-=64, write 7-->0 order
    }
}

__forceinline void chunk_4x8(int32_t*& dst_l, int32_t*& dst_r, int32_t*& sk, int32_t* end, int32_t piv, __m256i piv_vec)
{
    for ( ; sk != end; sk += 32)
    {
        __m256i window0 = _mm256_loadu_epi32(sk);                          
        __m256i window1 = _mm256_loadu_epi32(sk+8);
        __m256i window2 = _mm256_loadu_epi32(sk+16);
        __m256i window3 = _mm256_loadu_epi32(sk+24);
   
        __m256i cmpres0 = _mm256_cmpgt_epi32(piv_vec, window0);             
        __m256i cmpres1 = _mm256_cmpgt_epi32(piv_vec, window1);
        __m256i cmpres2 = _mm256_cmpgt_epi32(piv_vec, window2);
        __m256i cmpres3 = _mm256_cmpgt_epi32(piv_vec, window3);
     
        uint16_t mask0 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres0));  
        uint16_t mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres1));  
        uint16_t mask2 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres2));
        uint16_t mask3 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres3));
       
        __m256i idxs0 = permidxs8[mask0];                                   
        __m256i idxs1 = permidxs8[mask1];
        __m256i idxs2 = permidxs8[mask2];
        __m256i idxs3 = permidxs8[mask3];

        __m256i shuffled0 = _mm256_permutevar8x32_epi32(window0, idxs0);    
        __m256i shuffled1 = _mm256_permutevar8x32_epi32(window1, idxs1);    
        __m256i shuffled2 = _mm256_permutevar8x32_epi32(window2, idxs2);    
        __m256i shuffled3 = _mm256_permutevar8x32_epi32(window3, idxs3);
     
        uint8_t k0 = popcnts8[mask0];     // lookup table for __popcnt(mask), slightly faster.
        uint8_t k1 = popcnts8[mask1];     
        uint8_t k2 = popcnts8[mask2];     
        uint8_t k3 = popcnts8[mask3];     
    
    
        _mm256_storeu_epi32(dst_l, shuffled0);   dst_l += k0;               // free ymm 0..7
        _mm256_storeu_epi32(dst_l, shuffled1);   dst_l += k1;
        _mm256_storeu_epi32(dst_l, shuffled2);   dst_l += k2;
        _mm256_storeu_epi32(dst_l, shuffled3);   dst_l += k3;
   
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled0); dst_r += k0;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled1); dst_r += k1;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled2); dst_r += k2;
        dst_r -= 8; _mm256_storeu_epi32(dst_r, shuffled3); dst_r += k3;
    }
}

__forceinline void chunk_1x8(int32_t*& dst_l, int32_t*& dst_r, int32_t*& sk, int32_t* end, int32_t piv, __m256i piv_vec)
{
    for ( ; sk != end; sk += 8)
    {
        __m256i window = _mm256_loadu_epi32(sk);

        __m256i cmpres = _mm256_cmpgt_epi32(piv_vec, window);
        uint16_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));

        __m256i idxs = permidxs8[mask];
        __m256i shuffled = _mm256_permutevar8x32_epi32(window, idxs);
        uint8_t k = popcnts8[mask];     // lookup table for __popcnt(mask), slightly faster.
    
        _mm256_storeu_epi32(dst_l, shuffled);
        dst_l += k;

        dst_r -= 8;
        _mm256_storeu_epi32(dst_r, shuffled);
        dst_r += k;
    }
}

/// Simple bipartition (2-way, 1 pivot), input in *src with length sz.
/// Pivot must be passed into piv (as scalar) and piv_vec (as vector).
/// Less than partition is written into *dst_l in left to right order.
/// Greated than partition is written into *dst_r in right to left order.
__forceinline void bipartition_2_i32x8(int32_t*& dst_l, int32_t*& dst_r, int32_t* src, size_t sz, int32_t piv, __m256i piv_vec)
{
#define SZ(N) (sz-(sz%N))

    int32_t* sk = src;
    chunk_8x8(dst_l, dst_r, sk, src + SZ(64), piv, piv_vec);
    chunk_4x8(dst_l, dst_r, sk, src + SZ(32), piv, piv_vec);
    chunk_1x8(dst_l, dst_r, sk, src + SZ(8), piv, piv_vec);
   
    // handle leftover:
    for ( /* sk = end (already true) */; sk != (src+sz); sk++)
    {
        if (*sk <= piv) *(dst_l++) = *sk;
        else            *(--dst_r) = *sk;   // watch these pre vs. postfix
    }
}

__forceinline size_t bipartition_1_i32x8(int32_t* dst, int32_t* src, size_t sz, int32_t piv, __m256i piv_vec)
{
    int32_t* dst_l = dst, * dst_r = dst + sz;

    //inline of: bipartition_2_i32x8(dst_l, dst_r, src, sz, piv, piv_vec);
    bipartition_2_i32x8(dst_l, dst_r, src, sz, piv, piv_vec);

    return dst_l - dst;
}

struct Bucket {
    int32_t* begin;
    int32_t* end;

    __forceinline size_t size() { return end - begin; }
    __forceinline void print() 
    {
        std::cout << " [ ";
        for (int32_t* k = begin; k != end; k++)
            std::cout << *k << " ";
        std::cout << "]\n";
    }
};

struct Buckets {
    Bucket b0;
    Bucket b1;
    Bucket b2;
    Bucket b3;
    Bucket b4;
    Bucket b5;
    Bucket b6;
    Bucket b7;
};

enum bucket_alignment { BTK_ALIGN_LEFT, BTK_ALIGN_RIGHT };

__forceinline Bucket create_bucket(size_t sz, bucket_alignment align)
{
    Bucket b{};
    b.begin = new int32_t[sz];
    memset(b.begin, 0, sz);
    if (align) b.begin += sz;
    b.end = b.begin;
    return b;
}

/// Performs 8-way partition on the array beginning at *src, with size sz.
/// The 7 pivots are marked by p0..p7.
/// The results are emplaced in buckets, 
/// 
/// 
__forceinline void partition_8buckets_i32x8(int32_t* src, size_t sz, size_t chunk_sz, Buckets& bkts,
    int32_t p0, int32_t p1, int32_t p2, int32_t p3, int32_t p4, int32_t p5, int32_t p6)         // pivots
{
    // assume sz % CHUNK_SZ == 0

    int32_t* bf = new int[chunk_sz];    // = to memset?
    memset(bf, 0, chunk_sz);

    __m256i pv0 = _mm256_set1_epi32(p0),    // vectorized pivots
        pv1 = _mm256_set1_epi32(p1),
        pv2 = _mm256_set1_epi32(p2),
        pv3 = _mm256_set1_epi32(p3),
        pv4 = _mm256_set1_epi32(p4),
        pv5 = _mm256_set1_epi32(p5),
        pv6 = _mm256_set1_epi32(p6);
    
    int32_t* l, *r;

    for (int32_t* chunk = src; chunk != src + sz; chunk += chunk_sz)
    {
        // 2 way partition  src --> bf
        size_t k_ctr = bipartition_1_i32x8(bf, src, chunk_sz, p3, pv3); 

        // 4 way partition  bf --> src
        size_t k_left = bipartition_1_i32x8(src, bf, k_ctr, p1, pv1);
        size_t k_right = bipartition_1_i32x8(src + k_ctr, bf + k_ctr, chunk_sz - k_ctr, p5, pv5) + k_ctr;

#ifdef WRITE_TO_BUCKETS         // why is this sh!t slow
        // 8 way partition  src --> bkts
        bipartition_2_i32x8(bkts.b0.end, bkts.b1.begin, src, k_left, p0, pv0);
        bipartition_2_i32x8(bkts.b2.end, bkts.b3.begin, src + k_left, k_ctr - k_left, p2, pv2);
        bipartition_2_i32x8(bkts.b4.end, bkts.b5.begin, src + k_ctr, k_right - k_ctr, p4, pv4);
        bipartition_2_i32x8(bkts.b6.end, bkts.b7.begin, src + k_right, chunk_sz - k_right, p6, pv6);
#else 
        // 8 way partition  src --> bf 
        int32_t* _ = bf;
        int32_t* a = bf + k_left;
        int32_t* b = bf + k_ctr;
        int32_t* c = bf + k_right;
        int32_t* d = bf + chunk_sz;
        bipartition_2_i32x8(_, a, src, k_left, p0, pv0);
        bipartition_2_i32x8(a, b, src + k_left, k_ctr - k_left, p2, pv2);
        bipartition_2_i32x8(b, c, src + k_ctr, k_right - k_ctr, p4, pv4);
        bipartition_2_i32x8(c, d, src + k_right, chunk_sz - k_right, p6, pv6);
#endif
    }
}

void benchmark_buckets()
{
    SetThreadAffinityMask(GetCurrentThread(), 1 << 2);
    // generate dataset and fill with random data
    size_t chunk_sz = static_cast<size_t>(1) << 10,
                 sz = static_cast<size_t>(1) << 28; 
    int32_t* src = new int32_t[sz];

#define ARIF_DATAGEN
#ifdef ARIF_DATAGEN
    datagen::Writer<uint32_t> writer;
    writer.generate((uint32_t*)src, sz, datagen::MT, INT32_MAX, 0);
#else
    std::default_random_engine rng;
    rng.seed(std::random_device{}());
    std::uniform_int_distribution<int32_t> dist(0, INT32_MAX);
    for (int i = 0; i < sz; i++) src[i] = dist(rng);   
#endif

    std::cout << "Generated input dataset.\n";

    // construct buckets and emplace in struct
    // est. size of bucket = total size / 8 * 2 for safety
    // for the above input size this actualy equates to 
    size_t est_bkt_sz = (size_t)((sz / 8) * 1.2); 
    std::cout << "Alocating buckets of size " << est_bkt_sz << "\n";
    Buckets bkts = {
        create_bucket(est_bkt_sz, BTK_ALIGN_LEFT),
        create_bucket(est_bkt_sz, BTK_ALIGN_RIGHT),
        create_bucket(est_bkt_sz, BTK_ALIGN_LEFT),
        create_bucket(est_bkt_sz, BTK_ALIGN_RIGHT),
        create_bucket(est_bkt_sz, BTK_ALIGN_LEFT),
        create_bucket(est_bkt_sz, BTK_ALIGN_RIGHT),
        create_bucket(est_bkt_sz, BTK_ALIGN_LEFT),
        create_bucket(est_bkt_sz, BTK_ALIGN_RIGHT),
    };
    std::cout << "Allocated buckets.\n";

    int32_t p = INT32_MAX / 8;
    auto t_start = std::chrono::high_resolution_clock::now();                                               \
    partition_8buckets_i32x8(src, sz, chunk_sz, bkts, 
            p, 2*p, 3*p, 4*p, 5*p, 6*p, 7*p);
    
    auto t_end = std::chrono::high_resolution_clock::now();                                                 
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();            
    std::cout << "Partition completed.\n";
    std::cout << " took " << elapsed_time_ms << "ms to partition " << sz << " integers\n";           
    std::cout << " the partition rate = " << sz / (elapsed_time_ms * 1000000) << "b integers/s\n";   

    std::cout << " bucket 1 size = " << bkts.b0.size() << "\n";
    std::cout << " bucket 2 size = " << bkts.b1.size() << "\n";
    std::cout << " bucket 3 size = " << bkts.b2.size() << "\n";
    std::cout << " bucket 4 size = " << bkts.b3.size() << "\n";
    std::cout << " bucket 5 size = " << bkts.b4.size() << "\n";
    std::cout << " bucket 6 size = " << bkts.b5.size() << "\n";
    std::cout << " bucket 7 size = " << bkts.b6.size() << "\n";
    std::cout << " bucket 8 size = " << bkts.b7.size() << "\n";
}


void test_buckets(size_t sz)
{
    srand(1000);

    int32_t* src = new int[sz];
    // fill with random numbers [0, 80)
    for (int i = 0; i < sz; i++) src[i] = (rand() % 80);   
    // est. size of bucket = total size / 8 * 2 for safety
    size_t est_bkt_sz = (size_t)((sz / 8.0) * 2); 

    Buckets bkts = {
        create_bucket(est_bkt_sz, BTK_ALIGN_LEFT),
        create_bucket(est_bkt_sz, BTK_ALIGN_RIGHT),
        create_bucket(est_bkt_sz, BTK_ALIGN_LEFT),
        create_bucket(est_bkt_sz, BTK_ALIGN_RIGHT),
        create_bucket(est_bkt_sz, BTK_ALIGN_LEFT),
        create_bucket(est_bkt_sz, BTK_ALIGN_RIGHT),
        create_bucket(est_bkt_sz, BTK_ALIGN_LEFT),
        create_bucket(est_bkt_sz, BTK_ALIGN_RIGHT),
    };
    
    help::arrprint("in", src, sz);

    partition_8buckets_i32x8(src, sz, 64, bkts, 10, 20, 30, 40, 50, 60, 70);

    std::cout << bkts.b0.size() << ":";
    bkts.b0.print();

    std::cout << bkts.b1.size() << ":";
    bkts.b1.print();

    std::cout << bkts.b2.size() << ":";
    bkts.b2.print();

    std::cout << bkts.b3.size() << ":";
    bkts.b3.print();

    std::cout << bkts.b4.size() << ":";
    bkts.b4.print();

    std::cout << bkts.b5.size() << ":";
    bkts.b5.print();

    std::cout << bkts.b6.size() << ":";
    bkts.b6.print();

    std::cout << bkts.b7.size() << ":";
    bkts.b7.print();

}

void benchmark_bipartition()
{
    SetThreadAffinityMask(GetCurrentThread(), 1 << 2);
    // generate dataset and fill with random data
    size_t sz = static_cast<size_t>(1) << 10;
    size_t iters = 4000000;
    int32_t* dst = new int32_t[sz];
    memset(dst, 0, sz);
    int32_t* base = new int32_t[sz];

#define ARIF_DATEGEN
#ifdef ARIF_DATEGEN
    datagen::Writer<uint32_t> writer;
    writer.generate((uint32_t*)base, sz, datagen::CONST_RAND, INT32_MAX, 0);
#else
    std::default_random_engine rng;
    rng.seed(std::random_device{}());
    std::uniform_int_distribution<int32_t> dist(0, INT32_MAX);
    for (int i = 0; i < sz; i++) base[i] = dist(rng);   
#endif
    std::cout << "Generated input dataset.\n";
    int32_t* src = new int32_t[sz];

    int32_t p = INT32_MAX / 2;
    std::chrono::steady_clock::time_point t_start, t_end;
    double elapsed_time_ms = 0;
    for (int i = 0; i < iters; i++)
    {
        memcpy(src, base, sz);
        t_start = std::chrono::high_resolution_clock::now();                                               
        bipartition_1_i32x8(dst, src, sz, p, _mm256_set1_epi32(p));
        t_end = std::chrono::high_resolution_clock::now();                                                 
        elapsed_time_ms += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }
    std::cout << "Partition completed.\n";
    std::cout << " took " << elapsed_time_ms << "ms to partition " << (iters*sz) << " integers\n";           
    std::cout << " the partition rate = " << (iters*sz) / (elapsed_time_ms * 1000) << " m integers/s\n";   

}
