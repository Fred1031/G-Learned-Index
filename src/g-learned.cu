#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "pgm/pgm_index.hpp"
#include <assert.h>
#include <cuda.h>

#define TP uint32_t
const int SIZE = 1e7;
const int epsilon = 128; // space-time trade-off parameter

const int mod = 1 << 15;

namespace Zorder {
	unsigned int index(unsigned int x, unsigned int y) {
		unsigned int idx = 0;
		for (int i = 0; i < 15; i ++) {
			idx |= ((((x >> i) & 1) << 1) | ((y >> i) & 1) ) << (i << 1);
		}
		return idx;
	}

	std::pair<int,int> point(unsigned int idx) {
		unsigned int x = 0, y = 0;
		for (int i = 0; i < 15; i ++) {
			x |= ( (idx >> (i << 1 | 1)) & (0x1) ) << i;
			y |= ( (idx >> (i << 1)) & (0x1) ) << i;
		}

		return std::make_pair(x, y);
	}
}

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define SUB_EPS(x, epsilon) ((x) <= (epsilon) ? 0 : ((x) - (epsilon)))
#define ADD_EPS(x, epsilon, size) ((x) + (epsilon) + 2 >= (size) ? (size) : (x) + (epsilon) + 2)

int query[5000010];
int pos[5000010];
int lo[5000010];
int hi[5000010];
struct LEVELS {
	int levels_num;
	int levels_offsets[3];
} levels;
struct Segment {
	TP key;
	float slope;
	int32_t intercept;
};
struct SEGMENTS {
	int segments_num;
	Segment segments[3003530];
} segments;

int cpu_levels_offsets[10];
int cpu_segments[10000];


#define CALC(key, it) int64_t(segments->segments[(it)].slope * ((key) - segments->segments[(it)].key) + segments->segments[(it)].intercept)

void __global__ gpu_query(int first_key, LEVELS levels, SEGMENTS *segments, int N, int *g_query, int *g_pos, int *g_lo, int *g_hi) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		int key = g_query[i];
		key = MAX(key, first_key);
		
		int it = levels.levels_num - 1;

		int left = 0, right = levels.levels_offsets[1] - 1;
		while(left <= right) {
			int mid = (left + right) >> 1;
			if (segments->segments[mid].key <= key) {
				left = mid + 1;
				it = mid;
			} else 
				right = mid - 1;
		}


		int Next = ((*segments).segments[it + 1]).intercept;
		int pos = MIN(CALC(key, it), Next); // pos > 0 ? pos : 0
		pos = pos > 0 ? pos : 0;
		int lo = SUB_EPS(pos, epsilon);
		int hi = ADD_EPS(pos, epsilon, SIZE);
		g_pos[i] = pos;
		g_lo[i] = lo;
		g_hi[i] = hi;
	}
}

#include "util.h"
#include <string>
#include "timer.cuh"


int main() {
	freopen("random_1e7.in", "r", stdin);
    // Generate some random data
    std::vector<TP> data(0), datax(SIZE), datay(SIZE);
    std::generate(datax.begin(), datax.end(), std::rand);
    std::generate(datay.begin(), datay.end(), std::rand);

	for (int i = 0; i < SIZE; i ++) {
		std::cin >> datax[i] >> datay[i];
		data.push_back(Zorder::index(datax[i], datay[i]));
	}

    std::sort(data.begin(), data.end());

    // Construct the PGM-index
    pgm::PGMIndex<TP, epsilon> index(data);

	for (int i = 0; i < 5; i ++)
		std::cerr << index.levels_offsets[i] << std::endl;
	int levels_num = 2;
	levels.levels_num = levels_num;
	for (int i = 0; i < levels_num; i ++) {
		cpu_levels_offsets[i] = index.levels_offsets[i];
		levels.levels_offsets[i] = index.levels_offsets[i];
	}

	int segments_num = index.segments.size();
	segments.segments_num = segments_num;
	std::cerr << "seg_num = " << segments.segments_num << std::endl;
	for (int i = 0; i < segments_num; i ++) {
		cpu_segments[i] = index.segments[i];
		segments.segments[i].key = index.segments[i].key;
		segments.segments[i].slope = index.segments[i].slope;
		segments.segments[i].intercept = index.segments[i].intercept;
	}


    // Query the PGM-index
	freopen("query_1e7.in", "r", stdin);
	for (int i = 0; i < 10000; i ++) {
		int queryx, queryy;
		std::cin >> queryx >> queryy;
		query[i] = Zorder::index(queryx, queryy);
	}

	int N = 1 << 22;
	for (int i = 10000; i < N; i ++)
		query[i] = query[i - 10000];
	double START_CLOCK = clock();

	double tot_time = 0;
	double latency = 0;
	int batch_size = 1 * (1 << 22);
	for (int batch = 0; batch < N / batch_size; batch ++) {
		int m = sizeof(int) * batch_size;
		int m_segments = sizeof(segments);
		int *g_query;
		int *g_pos;
		int *g_lo;
		int *g_hi;
		SEGMENTS *g_segments;

		int block_size = 128;
		int grid_size = (batch_size - 1) / block_size + 1;
		cudaMalloc((void **)&g_query, m);
		cudaMalloc((void **)&g_pos, m);
		cudaMalloc((void **)&g_lo, m);
		cudaMalloc((void **)&g_hi, m);
		cudaMalloc((void **)&g_segments, m_segments);

		GpuTimer querytimer;
		querytimer.timerStart();

		cudaMemcpy(g_query, query + batch * batch_size, m, cudaMemcpyHostToDevice);
		cudaMemcpy(g_segments, &segments, m_segments, cudaMemcpyHostToDevice);

		gpu_query<<<grid_size, block_size>>>
		(
		 index.first_key,
		 levels,
		 g_segments,
		 N,
		 g_query,
		 g_pos,
		 g_lo, 
		 g_hi
		 );
		querytimer.timerStop();


		GpuTimer lower_bound_timer;
		lower_bound_timer.timerStart();

		cudaMemcpy(pos, g_pos, m, cudaMemcpyDeviceToHost);
		cudaMemcpy(lo, g_lo, m, cudaMemcpyDeviceToHost);
		cudaMemcpy(hi, g_hi, m, cudaMemcpyDeviceToHost);

		for (int i = 0; i < batch_size; i ++) {
			for (int j = lo[i]; j < hi[i]; j ++)
				if (query[i] == data[j])
					break;
		}
		lower_bound_timer.timerStop();
		// use clock
		double timer = querytimer.getNsElapsed() + lower_bound_timer.getNsElapsed();
		latency = MAX(latency, timer * 1e-6);
		tot_time += timer;

		cudaFree(&g_query);
		cudaFree(&g_pos);
		cudaFree(&g_lo);
		cudaFree(&g_hi);
		cudaFree(&g_segments);
	}
	std::cerr << "Throughput of G-PGM : "
		<< N * 1e6 / 1024 / 1024 / tot_time << " Mqueries / s" << std::endl;
	std::cerr << "Latency of G-PGM : "
		<< latency << " s" << std::endl;
	return 0;
}
