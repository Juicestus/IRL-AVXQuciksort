#pragma once

#include <unordered_set>
#include <fstream>
#include <execution>
#include "arif_common.h"

//#define PLD_L2
//#define PARETO_L2


namespace datagen {

	typedef enum {
		MT,
		ALL_SAME,
		PLD,
		PARETO_B2B,
		PARETO_SHUFF,
		PARETO_NONUNIFORM,
		RANDOM_PERM_N,
		SORTED,
		REV_SORTED,
		FIB,
		ALMOST_SORTED,
		CONST_RAND,
		NORMAL,
		UNIFORM_DBL,
		CRAND_GAUSSIAN,
		WORST_CASE,
		WORST_CASE2,
		PD,
		UD,
		U_SEQ,
		R_SEQ,
		WORST_CASE_BACKSCAN
	} WRITER_TYPE;

#define UNIFORM_64			0
#define UNIFORM_32			1
#define UNIFORM_N_BY_4		2
#define UNIFORM_N			3
#define UNIFORM_3N			4
#define UNIFORM_10N			5
#define UNIFORM_UPPER_N		6
#define UNIFORM_CUSTOM		7

#define MIN(x,y)	((x) < (y) ? (x) : (y))

	const char* uniform_ranges[] = { "2^64 - 1", "2^32 - 1", "n/4", "n-1", "3n-1", "10n-1", "[MAX - n + 1, MAX]", "Custom [l, m]" };
	const int uniform_ranges_count = 8;

	static const char* writer_names_[] = { "Mersenne-Twister", "All-Same", "PLD", "Pareto-b2b", "Pareto-shuffled", "Pareto-non-uniform",
									"Random-permutation", "Sorted-sequence", "Reverse-sorted", "Fibonacci-sequence",
									"Almost-sorted", "Constant-and-random", "Normal", "Uniform-double", "crand-Gaussian", "Worst-case", "Worst-case-2",
									"Predictable-dups", "Unpredictable-dups", "u-Sequential", "r-Sequential", "Worst-case-bscan" };

	static const char* PLD_path = "PLD-out-graph-1GB.adj";		// path on d5		"A:\\PLD-out-graph.adj
	const int writer_count_ = 22;

	template <typename Item, typename Key = Item>
	class Writer {
	public:
		void random_writer_mt(Item* A, ui64 n, ui64 m = ~0ull, ui64 l = 0) {
			// key only
			if constexpr (std::is_same<Item, Key>::value) {
				if constexpr (std::is_same<Item, ui>::value) {
					std::mt19937 g;
					std::uniform_int_distribution<Item> d(l, MIN(m, UINT32_MAX));
					FOR(i, n, 1) A[i] = d(g);
				}
				else if constexpr (std::is_same<Item, ui64>::value) {
					std::mt19937_64 g;
					std::uniform_int_distribution<Item> d(l, m);
					FOR(i, n, 1) A[i] = d(g);
				}
				else
					ReportError("Type not supported");
			}
			// else -- key-value pair
			else {
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
					std::mt19937 g;
					std::uniform_int_distribution<ui> d(l, MIN(m, UINT32_MAX));
					FOR(i, n, 1) {
						ui64 kv = d(g);
						kv = (kv << 32) | i;		// add value to LSB
						A[i] = kv;
					}
				}
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
					std::mt19937_64 g;
					std::uniform_int_distribution<ui64> d(l, m);
					FOR(i, n, 1) {
						KeyValue<ui64, ui64> kv;
						kv.key = d(g);
						kv.value = i;
						A[i] = kv;
					}
				}
				else
					ReportError("Type not supported");
			}
		}

		void pareto_writer(Item* A, ui64 n, WRITER_TYPE type) {
			ui64 a = 6364136223846793005, c = 1442695040888963407, x = 1;
			double ED = 20;
			double alpha = 1, beta = 7;
			ui64 sum = 0, Items = 0, y = 889;
			ui64 maxF = 0;
			for (ui64 i = 0; i < n; ) {
				x = x * a + c;
				y = y * a + c;

				// generate frequency from the Pareto distribution with alpha=1; otherwise, the generator gets slow
				double u = (double)y / ((double)(1LLU << 63) * 2);			// uniform [0,1]
				ui64 f = MIN(ceil(beta * (1 / (1 - u) - 1)), 10000);		// rounded-up Pareto
				if (type == WRITER_TYPE::PARETO_B2B || type == WRITER_TYPE::PARETO_SHUFF) {
					if (i + f < n) {
						FOR(j, f, 1) {
							if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
								ui64 kv = x;
								kv = (kv << 32) | i;
								A[i + j] = kv;
							}
							else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
								KeyValue<ui64, ui64> kv;
								kv.key = x;
								kv.value = i;
								A[i + j] = kv;
							}
							else A[i + j] = x;
						}
						i += f;
					}
					else if (i + 10 >= n) {
						for (; i < n; ++i) {
							if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
								ui64 kv = x;
								kv = (kv << 32) | i;
								A[i] = kv;
							}
							else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
								KeyValue<ui64, ui64> kv;
								kv.key = x;
								kv.value = i;
								A[i] = kv;
							}
							else A[i] = x;
						}
					}
				}
				else if (type == WRITER_TYPE::PARETO_NONUNIFORM) {
					if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
						ui64 kv = f;
						kv = (kv << 32) | i;
						A[i] = kv;
					}
					else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
						KeyValue<ui64, ui64> kv;
						kv.key = f;
						kv.value = i;
						A[i] = kv;
					}
					else A[i] = f;
					i++;
				}
			}

			if (type == WRITER_TYPE::PARETO_SHUFF) {
				//printf("> Shuffling ... ");
				std::random_device rd;
				std::mt19937_64 g(rd());
				std::shuffle(A, A + n, g);
				//printf("done\n");
			}
		}

		void all_same(Item* A, ui64 n) {
			std::mt19937_64 g;
			std::uniform_int_distribution<ui64> d;
			ui64 x = d(g);

			FOR(i, n, 1) {
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
					ui64 kv = x;
					kv = (kv << 32) | i;
					A[i] = kv;
				}
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
					KeyValue<ui64, ui64> kv;
					kv.key = x;
					kv.value = i;
					A[i] = kv;
				}
				else A[i] = x;
			}
		}

		void u_sequential(Item* A, ui64 n) {
			FOR(i, n, 1) {
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
					ui64 kv = i;
					kv = (kv << 32) | i;
					A[i] = kv;
				}
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
					KeyValue<ui64, ui64> kv;
					kv.key = i;
					kv.value = i;
					A[i] = kv;
				}
				else A[i] = i;
			}
		}

		// rSEQ: force all keys to go to same bucket by reducing the sequence to be contained only in the last KEY_BITS - 8 bits
		void r_sequential(Item* A, ui64 n) {
			Item k;
			if constexpr (std::is_same<Item, ui>::value || std::is_same<Item, ui64>::value)
				k = 0;

			FOR(i, n, 1) {
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
					ui64 kv = i;
					kv = (kv << 32) | i;
					A[i] = kv;
				}
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
					KeyValue<ui64, ui64> kv;
					kv.key = i;
					kv.value = i;
					A[i] = kv;
				}
				else A[i] = k | (i & (1LLU << ((sizeof(Item) << 3) - 1)));
			}
		}

		void sorted(Item* A, ui64 n) {
			random_writer_mt(A, n);
			//std::sort(A, A + n);
			std::sort(std::execution::par_unseq, A, A + n);
		}

		void reverse_sorted(Item* A, ui64 n) {
			random_writer_mt(A, n);
			//std::sort(A, A + n, std::greater<>());
			std::sort(std::execution::par_unseq, A, A + n, std::greater<>());
		}

		// random permutation of numbers from 0 to n - 1
		void random_perm(Item* A, ui64 n) {
			FOR(i, n, 1) {
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
					ui64 kv = i;
					kv = (kv << 32) | i;
					A[i] = kv;
				}
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
					KeyValue<ui64, ui64> kv;
					kv.key = i;
					kv.value = i;
					A[i] = kv;
				}
				else A[i] = i;
			}
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(A, A + n, g);
		}

		void fibonacci(Item* A, ui64 n) {
			ui64 a = 0, b = 1, c;
			if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
				ui64 kv = 0; A[0] = kv;
				kv = (1LLU << 32) | 1; A[1] = kv;
			}
			else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
				KeyValue<ui64, ui64> kv;
				kv.key = 0; kv.value = 0; A[0] = kv;
				kv.key = 1; kv.value = 1; A[1] = kv;
			}
			else {
				A[0] = 0; A[1] = 1;
			}

			ui64 i = 2;
			while (i < n) {
				c = a + b;
				if (c < b) {	// overflow
					a = 0; b = 1;
					if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
						ui64 kv = (0LLU << 32) | i;
						A[i] = kv;
					}
					else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
						KeyValue<ui64, ui64> kv;
						kv.key = 0;
						kv.value = i;
						A[i] = kv;
					}
					else A[i] = 0;
					i++;
					if (i < n) {
						if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
							ui64 kv = (1LLU << 32) | i;
							A[i] = kv;
						}
						else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
							KeyValue<ui64, ui64> kv;
							kv.key = 1;
							kv.value = i;
							A[i] = kv;
						}
						else A[i] = 1;
						i++;
					}
				}
				else {
					a = b;
					b = c;
					if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
						ui64 kv = (b << 32) | i;
						A[i] = kv;
					}
					else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
						KeyValue<ui64, ui64> kv;
						kv.key = b;
						kv.value = i;
						A[i] = kv;
					}
					else A[i] = b;
					i++;
				}
			}
		}

		// set every 7 key of a sorted sequence to MAX
		void almost_sorted(Item* A, ui64 n) {
			sorted(A, n);
			FOR(i, n, 7) {
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) A[i] = (UINT32_MAX << 32) | i;
				else if constexpr (std::is_same<Item, ui64>::value) A[i] = UINT64_MAX;
				else if constexpr (std::is_same<Item, ui>::value) A[i] = UINT32_MAX;
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) A[i].key = UINT64_MAX;
			}
		}

		// 1/3rds of keys const, 2/3rds random; then shuffled
		void const_random(Item* A, ui64 n) {
			std::mt19937_64 gen;
			std::uniform_int_distribution<ui64> dis(0, ~0ull);

			// write 2/3rds random keys
			ui64 i = 0;
			for (; i < 2 * n / 3; ++i) {
				ui64 x = dis(gen);
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
					ui64 kv = (x << 32) | i;
					A[i] = kv;
				}
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
					KeyValue<ui64, ui64> kv;
					kv.key = x; kv.value = i; A[i] = kv;
				}
				else A[i] = x;
			}

			// write 1/3rds const keys
			ui64 x = dis(gen);
			for (; i < n; ++i) {
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
					ui64 kv = (x << 32) | i;
					A[i] = kv;
				}
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
					KeyValue<ui64, ui64> kv;
					kv.key = x; kv.value = i; A[i] = kv;
				}
				else A[i] = x;
			}
			std::random_device rd;
			std::mt19937_64 g(rd());
			std::shuffle(A, A + n, g);
		}

		// normal distribution w/ miu = max/2 and sigma = max/6
		void normal(Item* A, ui64 n, ui64 miu = (UINT64_MAX >> 1), ui64 sigma = (UINT64_MAX >> 1) / 3) {
			//printf("> Mean: %llu, std. dev: %llu\n", miu, sigma);
			if constexpr (std::is_same<Item, ui>::value) {
				miu = UINT32_MAX >> 1;
				sigma = (UINT32_MAX >> 1) / 3;
				std::mt19937 gen;
				std::normal_distribution<> dis(miu, sigma);

				FOR(i, n, 1)
					A[i] = std::round(dis(gen));
			}
			else if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
				miu = UINT32_MAX >> 1;
				sigma = (UINT32_MAX >> 1) / 3;
				std::mt19937 gen;
				std::normal_distribution<> dis(miu, sigma);

				FOR(i, n, 1) {
					ui64 key = std::round(dis(gen));
					A[i] = (key << 32) | i;
				}
			}
			else if constexpr (std::is_same<Item, ui64>::value) {
				miu = UINT64_MAX >> 1;
				sigma = (UINT64_MAX >> 1) / 3;
				std::mt19937_64 gen;
				std::normal_distribution<> dis(miu, sigma);

				FOR(i, n, 1)
					A[i] = std::round(dis(gen));
			}

			else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
				miu = UINT64_MAX >> 1;
				sigma = (UINT64_MAX >> 1) / 3;
				std::mt19937_64 gen;
				std::normal_distribution<> dis(miu, sigma);

				FOR(i, n, 1) {
					KeyValue<ui64, ui64> kv;
					kv.key = std::round(dis(gen));; kv.value = i; A[i] = kv;
				}
			}
			else
				ReportError("Type not supported");
		}

		void crand_gaussian(Item* A, ui64 n) {
			std::mt19937_64 gen;
			std::uniform_int_distribution<ui64> dis(0, ~0ull);

			FOR(i, n, 1) {
				ui64 a = dis(gen); ui64 b = dis(gen); ui64 c = dis(gen); ui64 d = dis(gen);
				ui64 x = (a >> 2) + (b >> 2) + (c >> 2) + (d >> 2) + (a & 4 + b & 4 + c & 4 + d & 4) / 4;
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
					A[i] = (x << 32) | i;
				}
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
					KeyValue<ui64, ui64> kv;
					kv.key = x; kv.value = i; A[i] = kv;
				}
				else
					A[i] = x;
			}
		}

		void uniform_double(Item* A, ui64 n) {
			if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
				std::uniform_real_distribution<float> dis(0.0, FLT_MAX);
				std::mt19937 gen;
				double* p = (double*)A;
				FOR(i, n, 1)
					p[i] = dis(gen);
				FOR(i, n, 1) A[i] = (A[i] & 0xFFFFFFFF00000000) | i;
			}
			else if constexpr (std::is_same<Item, ui>::value) {
				float* p = (float*)A;
				std::uniform_real_distribution<float> dis(0.0, FLT_MAX);
				std::mt19937 gen;
				FOR(i, n, 1)
					p[i] = dis(gen);
			}

			else if constexpr (std::is_same<Item, ui64>::value) {
				double* p = (double*)A;
				std::uniform_real_distribution<double> dis(0.0, DBL_MAX);
				std::mt19937_64 gen;
				FOR(i, n, 1)
					p[i] = dis(gen);
			}
			else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
				std::uniform_real_distribution<double> dis(0.0, DBL_MAX);
				std::mt19937_64 gen;
				FOR(i, n, 1) {
					KeyValue<ui64, ui64> kv;
					kv.key = dis(gen); kv.value = i; A[i] = kv;
				}
			}

		}

		// NOTE: worst case inputs not updated for key-value pairs
		void worst_case(Item* A, ui64 n) {
			/*
			the idea is to keep the uniform distribution on first two levels 8+8 bits
			after that, all items should go to the same bucket but the Items should not be
			the same; so the last few bits should vary
			Q. should we enforce SN30/31 or make the group large enough to do another level of split?
			*/
			if constexpr (std::is_same<Item, ui64>::value || std::is_same<Item, ui>::value) {
				std::mt19937_64 gen;
				std::uniform_int_distribution<UINT64> dis(0, UINT64_MAX);
				std::mt19937_64 gen2;
				std::uniform_int_distribution<UINT64> dis2(0, 255);
				const ui base_sort = 32;		// Vortex
				//const ui base_sort = 512;		// RADULS
				//const ui base_sort = 256;


				Item* p = A;
				ui steps = sizeof(Item);					// if 8 bytes -> 5 steps; 4 bytes -> 1 step
				ui shift_begin = (sizeof(Item) << 3);		// first two levels keep random

				ui64 i = 0;
				for (; i <= n - (base_sort + steps); i += (base_sort + steps)) {
					UINT64 x = dis(gen);
					int k = 0;
					for (; k < base_sort; ++k) {							// base_sort keys going to same buckets
						p[i + k] = x + k;
					}
					for (int j = shift_begin; j >= 8; j -= 8) {
						UINT64 mask = 255LLU << j;
						UINT64 x1 = (x & ~mask) | ((x & mask) ^ mask);		// add a key per level to prohibit all keys going to same bucket
						if (i + k >= n) return;
						p[i + k] = x1;
						k++;
					}
				}
				while (i < n) {
					ui64 x = dis(gen);
					p[i] = x;
					i++;
				}
			}
			else ReportError("Type not supported");
		}

		void worst_case2(Item* A, ui64 n) {
			if constexpr (std::is_same<Item, ui64>::value) {
				std::mt19937_64 gen;
				std::uniform_int_distribution<ui64> dis(0, UINT64_MAX);

				std::mt19937_64 gen2;
				std::uniform_int_distribution<ui64> dis2(0, 255);

				for (ui64 i = 0; i <= n - 37; i += 37) {
					ui64 x = dis(gen);
					for (int j = 40; j >= 8; j -= 8) {
						ui64 mask = 255LLU << j;
						ui64 y = dis2(gen2);
						x = (x & ~mask) | (y << j);
					}
					int k = 0;
					for (; k < 32; ++k)
						A[i + k] = x + k;
					for (int j = 40; j >= 8; j -= 8) {
						ui64 mask = 255LLU << j;
						ui64 x1 = (x & ~mask) | ((x & mask) ^ mask);
						A[i + k++] = x1;
					}
				}
			}
			else ReportError("Type not supported");

		}

		void worst_case_bscan(Item* A, ui64 n) {
			if constexpr (std::is_same<Item, ui>::value || std::is_same<Item, ui64>::value) {
				const ui shift = 8;
				ui size_bytes = sizeof(Item);
				ui64 mx = (1LLU << (size_bytes << 3)) - 1;

				ui64 i = 0;
				//FFFFFFF.., 00FFFFFF.., 0000FFFF...
				while (mx) {
					A[i++] = mx;
					mx >>= shift;
				}
				// fill the rest with zeros
				memset(A + i, 0, (n - i) * size_bytes);
			}
		}

		void pld(Item* A, ui64& n, int level) {
			ifstream inf(PLD_path, ios::in | ios::binary);
			ui64 x, i = 0;
			if (inf.is_open()) {
				if (level == 0) {
					while (inf.read(reinterpret_cast<char*>(&x), sizeof(ui64)) && i < n) {
						if constexpr (std::is_same<Item, KeyValue<ui, ui>>::value) {
							KeyValue<ui, ui> kv;
							kv.key = x; kv.value = i; A[i] = kv;
						}
						else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
							KeyValue<ui64, ui64> kv;
							kv.key = x; kv.value = i; A[i] = kv;
						}
						else
							A[i] = x;
						i++;
						//printf("%lu\n", i);
					}
				}
				else {
					ui64 targetItems = n;
					int shift = (64 - (level << 3));		// we want Items where first 8 * levels bits are zeros
					while (inf.read(reinterpret_cast<char*>(&x), sizeof(ui64)) && i < targetItems) {
						if ((x >> shift) == 0) {
							if constexpr (std::is_same<Item, KeyValue<ui, ui>>::value) {
								KeyValue<ui, ui> kv;
								kv.key = x; kv.value = i; A[i] = kv;
							}
							else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
								KeyValue<ui64, ui64> kv;
								kv.key = x; kv.value = i; A[i] = kv;
							}
							else
								A[i] = x;
							i++;
						}
					}
					n = i - (i & 1);
				}
			}
			else
				ReportError("PLD file path not correct");
			inf.close();
		}

		void predictable_dups(Item* A, ui64 n, ui64 rep) {
			ui64 a = 6364136223846793005, c = 1442695040888963407, x = 1;
			uint64_t i = 0;
			for (; i <= n - rep; i += rep) {
				x = x * a + c;
				FOR(j, rep, 1) {
					if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
						A[i + j] = (x << 32) | (i + j);
					}
					else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
						KeyValue<ui64, ui64> kv;
						kv.key = x; kv.value = i + j; A[i + j] = kv;
					}
					else A[i + j] = x;
				}
			}
			for (; i < n; ++i) {
				x = x * a + c;
				if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
					A[i] = (x << 32) | i;
				}
				else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
					KeyValue<ui64, ui64> kv;
					kv.key = x; kv.value = i; A[i] = kv;
				}
				else A[i] = x;
			}
		}

		void unpredictable_dups(Item* A, ui64 n, ui64 X1, ui64 X2) {
			srand(time(NULL));
			ui64 a = 6364136223846793005, c = 1442695040888963407, x = 1;
			uint64_t i = 0;
			for (; i < n;) {
				double r = rand() * 1.0 / RAND_MAX;
				x = x * a + c;
				if (r > 0.5 && i + X1 < n) {
					FOR(j, X1, 1) {
						if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
							A[i + j] = (x << 32) | (i + j);
						}
						else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
							KeyValue<ui64, ui64> kv;
							kv.key = x; kv.value = i + j; A[i + j] = kv;
						}
						else A[i + j] = x;
					}
					i += X1;
				}
				else if (i + X2 < n) {
					FOR(j, X2, 1) {
						if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
							A[i + j] = (x << 32) | (i + j);
						}
						else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
							KeyValue<ui64, ui64> kv;
							kv.key = x; kv.value = i + j; A[i + j] = kv;
						}
						else A[i + j] = x;
					}
					i += X2;
				}
				else {
					for (; i < n; ++i) {
						if constexpr (std::is_same<Item, ui64>::value && std::is_same<Key, ui>::value) {
							A[i] = (x << 32) | i;
						}
						else if constexpr (std::is_same<Item, KeyValue<ui64, ui64>>::value) {
							KeyValue<ui64, ui64> kv;
							kv.key = x; kv.value = i; A[i] = kv;
						}
						else A[i] = x;
					}
				}
			}
		}

		void usage() {
			printf("> Input types:\n");
			for (int i = 1; i < writer_count_; ++i) {
				printf("%2d %s\n", i, writer_names_[i]);
				if (i == MT) {
					for (int j = 0; j < uniform_ranges_count; ++j) {
						printf("    %2d %s\n", j, uniform_ranges[j]);
					}
				}
			}
		}

		// write random keys to random n_buckets buckets
		void random_buckets(Item* A, ui64 n, ui n_buckets, ui tot_buckets) {
#define ROUND_ROBIN
			/*A[0] = 0;
			FOR_INIT(i, 1, n, 1) {
				if ((i & 63) == 0) A[i] = 1;
				else if ((i & 63) == 63) A[i] = 0;
				else A[i] = (i & 1);

				A[i] = A[i] << 56;
			}
			return;*/

			std::mt19937_64 g;
			std::uniform_int_distribution<ui64> d;

			// using 0 through n_buckets - 1 
			FOR(i, n, 1) {
#ifdef ROUND_ROBIN
				A[i] = (i % n_buckets) << 56;		// round-robin 
#else 
				A[i] = (d(g) % n_buckets) << 56;	// random	
#endif 
			}
#undef ROUND_ROBIN
			return;

			// using random n_buckets buckets
			// generate random keys to use
			Item* keys = new Item[n_buckets];
			std::unordered_set<ui> S;
			FOR(i, n_buckets, 1) {
				while (1) {
					ui64 k = d(g);
					auto p = S.insert(k >> 56);
					if (p.second) {
						keys[i] = k;
						break;
					}
				}
			}
			// now distribute the keys
			std::mt19937 g2;
			std::uniform_int_distribution<ui> d2(0, n_buckets - 1);
			FOR(i, n, 1) {
				ui idx = d2(g2);
				A[i] = keys[idx];
			}

			delete[] keys;

		}

		// l, m is the range for MT random Items only
		void generate(Item* A, ui64& n, WRITER_TYPE type, ui64 m = ~0ull, ui64 l = 0, ui64 rep = 1, ui64 X1 = 1, ui64 X2 = 1) {
			//printf("> Writer: %s ... ", writer_names_[type]);
			if (type == MT)
				random_writer_mt(A, n, m, l);
			else if (type == RANDOM_PERM_N)
				random_perm(A, n);
			else if (type == SORTED)
				sorted(A, n);
			else if (type == REV_SORTED)
				reverse_sorted(A, n);
			else if (type == FIB)
				fibonacci(A, n);
			else if (type == ALMOST_SORTED)
				almost_sorted(A, n);
			else if (type == CONST_RAND)
				const_random(A, n);
			else if (type == NORMAL)
				normal(A, n);
			else if (type == CRAND_GAUSSIAN)
				crand_gaussian(A, n);
			else if (type == UNIFORM_DBL)
				uniform_double(A, n);
			else if (type == WORST_CASE)
				worst_case(A, n);
			else if (type == WORST_CASE2)
				worst_case2(A, n);
			else if (type == ALL_SAME)
				all_same(A, n);
			else if (type == PLD)
				pld(A, n, rep);
			else if (type == U_SEQ)
				u_sequential(A, n);
			else if (type == R_SEQ)
				r_sequential(A, n);
			else if (type == PARETO_SHUFF || type == PARETO_B2B || type == PARETO_NONUNIFORM)
				pareto_writer(A, n, type);
			else if (type == PD) {
				predictable_dups(A, n, rep);

				// to random buckets
				/*ui n_buckets = 2;
				PRINT("Buckets? "); scanf("%lu", &n_buckets);
				ui tot_buckets = 256;
				random_buckets(A, n, n_buckets, tot_buckets);*/
			}
			else if (type == UD)
				unpredictable_dups(A, n, X1, X2);
			else if (type == WORST_CASE_BACKSCAN)
				worst_case_bscan(A, n);

			//printf("done\n");

			/*double* hist = new double[256];
			utils::histogram(A, n, 8, 48, hist);
			FOR(i, 256, 1) printf("\n%3d: %.2f", i, hist[i]); printf("\n");
			delete hist;*/

			/*printf("> Printing first few items to verify input:\n");
			FOR(i, 20, 1) printf("%llX ", A[i]); printf("\n");*/

			//int print_items = 100;
			//for (int i = 0; i < print_items; ++i)
			//	printf("%llx%c", A[i], i == (print_items - 1) ? '\n' : ' ');
			//	// for PARETO_L1
			//	//printf("%llu%c", A[i] >> 6, i == (print_items - 1) ? '\n' : ' ');
			//

			//RunlengthHist(A, n, 8, 6);
		}
	};
};

