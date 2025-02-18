# IRL AVX Quicksort 

Experiments/benchmarks for AVX2 Quicksort, affiliated with [IRL @ Texas A&M University](http://irl.cs.tamu.edu/).

Reach out to Justus at <justus@tamu.edu> or <jus@justusl.com>.

## Results

#ifdef benchmark_bipartition    // ~13.5 b ints/s
    BENCHMARK(simple_bipartition_i32x8(dst, src, sz, INT32_MAX / 2));
#endif
   
#ifdef benchmark_4partition     //  ~6.5 b ints/s
    BENCHMARK(simple_4partition_i32x8(dst, src, sz, std::make_tuple(       
        INT32_MAX / 4, INT32_MAX / 2, 3 * INT32_MAX / 4)));
   
8 partition     //  ~3.3 b ints/s
