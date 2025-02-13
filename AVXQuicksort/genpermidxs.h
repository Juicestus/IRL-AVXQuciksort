#pragma once

#include <bitset>
#include <string>
#include <fstream>
#include <iostream>
#include <smmintrin.h>
#include <immintrin.h>



__m256i permidx8(uint8_t mask)
{
    uint32_t a = 0, b = 7;
    uint32_t idxs[8];
	for (uint8_t j = 0; j < 8; j++) {
        uint32_t k = ((mask >> j) & 1) ? a++ : b--;
        idxs[k] = j;
	}
    return _mm256_loadu_epi32(idxs);
}

///
/// Creates a header file "permidxs8.h" which contains the mapping 
/// of masks -> permutations for 8 bit registers using mm setr.
///
/// This function is kind of a mess -- probably would've been better
/// to be a Python script... oh well
/// 
void dump_permidxs_mask8()
{
    const size_t size = (1 << 8);

    std::ofstream fs("permidxs8.h");
    fs << "#pragma once\n\n";
    fs << "#include <smmintrin.h>\n";
    fs << "#include <immintrin.h>\n";
    fs << "\nstatic const __m256i permidxs8[" << size << "] = {\n";

    size_t i, j, a, b, k;
    uint32_t idxs[8];

    for (i = 0; i < size; i++)
    {
        for (j = 0, a = 0, b = 7; j < 8; j++) {
            k = ((i >> j) & 1) ? a++ : b--;
            idxs[k] = j;
        }

        fs << "    _mm256_setr_epi32(";
        for (j = 0; j < 8; j++) {
            if (j > 0) fs << ", ";
            fs << idxs[j];
        }
        fs << "),\n";

    }
    fs << "};\n\n";
}


