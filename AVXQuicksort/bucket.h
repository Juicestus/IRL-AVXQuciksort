#pragma once

#include "partition.h"

#include <cstdint>
#include <intrin.h>
#include <smmintrin.h>
#include <immintrin.h> 
#include <tuple>
#include <vector>

/// 
/// Performs 8-way partition on the array beginning at
/// by *src with size sz, around the pivots in pivs.
/// 
/// 
/// 
void simple_partition_8buckets_i32x8(int32_t* src, size_t sz, tuple7<int32_t> pivs, tuple8<int32_t*> buckets)
{

}

