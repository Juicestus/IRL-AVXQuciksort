#pragma once

// common.h
#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>
#include <iostream>
#include <time.h>
#include <stdexcept>
#include <stdio.h>

#include <random>
#include <stdexcept>
#include <thread>
#include <stack>
#include <queue>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include <filesystem>
#include <cassert>
using namespace std;

#include <immintrin.h>

#define MAX_PRINTOUT	1024
#define PRINT(fmt, ...) { char buf_PRINT[MAX_PRINTOUT] = "%s: "; strcat_s(buf_PRINT, MAX_PRINTOUT, fmt); printf (buf_PRINT, __FUNCTION__, ##__VA_ARGS__); }
#define ReportError(fmt, ...) { PRINT(fmt, ##__VA_ARGS__); \
								exit(-1); }

typedef unsigned __int64 uint64;
typedef __int64 int64;

using namespace std;
using namespace std::chrono;
using hrc = high_resolution_clock;

// typedef Item
typedef int64_t  i64;
typedef uint64_t ui64;
typedef uint32_t ui;
typedef uint8_t	uchar;
typedef uint16_t ushort;

typedef __m256i avx2;

#pragma pack(push, 1)
template <typename Keytype, typename Valuetype>
struct KeyValue {
	Keytype key;
	Valuetype value;
	bool operator <(const KeyValue& kv) const {
		return key < kv.key;
	}
	bool operator >(const KeyValue& kv) const {
		return key > kv.key;
	}
	bool operator !=(const KeyValue& kv) const {
		return key != kv.key;
	}
	bool operator <=(const KeyValue& kv) const {
		return key <= kv.key;
	}
};
#pragma pack(pop)
template struct KeyValue<ui, ui>;
template struct KeyValue<ui64, ui64>;

#define MIN(x, y)				((x)<(y)?(x):(y))
#define MAX(x, y)				((x)<(y)?(y):(x)) 
#define FOR(i,n,k)				for (ui64 (i) = 0; (i) < (n); (i)+=(k)) 
#define FOR_INIT(i, init, n, k)	for (ui64 (i) = (init); (i) < (n); (i) += (k)) 
#define PRINT_ARR(arr, n)		{ FOR((i), (n), 1) printf("%lX ", (arr)[(i)]); printf("\n"); }
#define PRINT_ARR64(arr, n)		{ FOR((i), (n), 1) printf("%llX ", ((ui64*)arr)[(i)]); printf("\n"); }
#define PRINT_DASH(n)			{ FOR(i, (n), 1) printf("-"); printf("\n"); }
#define ELAPSED(st, en)			( duration_cast<duration<double>>(en - st).count() )
#define ELAPSED_MS(st, en)		( duration_cast<duration<double, std::milli>>(en - st).count() )
#define ELAPSED_NS(st, en)		( duration_cast<duration<double, std::nano>>(en - st).count() )
#define NOINLINE				__declspec(noinline)
#define KB(x)					(x << 10)
#define MB(x)					(x << 20)
#define GB(x)					(x << 30)
#define HERE(x)					printf("Here %3lu\n", (x));
#define MAX_PATH_LEN			512
#define MAX_PRINTOUT			1024
#define PRINT(fmt, ...)			{ char buf_PRINT[MAX_PRINTOUT] = "%s: "; strcat_s(buf_PRINT, MAX_PRINTOUT, fmt); printf (buf_PRINT, __FUNCTION__, ##__VA_ARGS__); }
#define ReportError(fmt, ...)	{ PRINT(fmt, ##__VA_ARGS__); getchar(); exit(-1); }
#define ROUND_UP(x,R)			((  ((x) + (R)-1) / (R) ) * (R) )
#define ROUND_DOWN(x, s)		((x) & ~((s)-1))
#define NINE_BIT_MASK			((1<<9) - 1)
#define GETCHAR					{ printf("Program finished, press enter to exit"); getchar(); }

#define LOAD(rg, ptr)			{ rg = *(ptr); }
#define STORE(rg, ptr)			{ *(ptr) = rg; }
#define VALLOC(sz)				(VirtualAlloc(NULL, (sz), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
#define VFREE(ptr)				(VirtualFree((ptr), 0, MEM_RELEASE))
