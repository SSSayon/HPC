// #include <chrono>
// #include <cstdio>

#include "apsp.h"

// namespace ch = std::chrono;

#define M 100001
#define BLOCK_SIZE 32
#define SPAN 4
/* 
`FIXED_SPAN` is NOT a hyper-parameter

to make full use of the shared memory, the equation below need to be met: 

2 * FIXED_SPAN * BLOCK_SIZE * BLOCK_SIZE * sizeof(int) B = 48 KB
*/
#define FIXED_SPAN 6 * 1024 / (BLOCK_SIZE * BLOCK_SIZE)

namespace {

__global__ void phase1(int n, int p, int *graph) {
    __shared__ int shared[BLOCK_SIZE * BLOCK_SIZE];
    int tmp = M;

    int i = p * BLOCK_SIZE + threadIdx.y;
    int j = p * BLOCK_SIZE + threadIdx.x;
    int local_pos = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    if (i < n && j < n) {
        shared[local_pos] = graph[i * n + j];
    } else {
        shared[local_pos] = M; // pay attention to the boundary
    }

    __syncthreads();

    if (i < n && j < n) {

        // tmp = shared[local_pos];
        // tmp = M;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // tmp = shared[threadIdx.y * BLOCK_SIZE + k] + shared[k * BLOCK_SIZE + threadIdx.x];
            tmp = min(tmp, shared[threadIdx.y * BLOCK_SIZE + k] + shared[k * BLOCK_SIZE + threadIdx.x]);
            // __syncthreads();
            // shared[local_pos] = min(shared[local_pos], tmp);
            // __syncthreads();
            // NOTE: use `min` (provided by CUDA itself) instead of `std::min`
        }
        // __syncthreads();
        // if (i < n && j < n)
            graph[i * n + j] = tmp;
    }
} 

__global__ void phase2(int n, int p, int *graph) {
    __shared__ int center[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int shared[SPAN][BLOCK_SIZE * BLOCK_SIZE];
    int tmp;

    int i = p * BLOCK_SIZE + threadIdx.y;
    int j = p * BLOCK_SIZE + threadIdx.x;
    int local_pos = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    if (i < n && j < n) {
        center[local_pos] = graph[i * n + j];
    } else {
        center[local_pos] = M;
    }

    if (blockIdx.y == 0) { // row

        int col_j = blockIdx.x * SPAN * BLOCK_SIZE + threadIdx.x;
        for (int span = 0; span < SPAN; ++span) {
            if (i < n && col_j < n) {
                shared[span][local_pos] = graph[i * n + col_j];
            } else {
                shared[span][local_pos] = M;
            }  
            col_j += BLOCK_SIZE;
        }

        __syncthreads();

        if (i >= n) return;
        col_j = blockIdx.x * SPAN * BLOCK_SIZE + threadIdx.x;
        for (int span = 0; span < SPAN; ++span) {
            if (col_j >= n) return;

            // tmp = shared[span][local_pos];
            tmp = M;
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                // tmp = center[threadIdx.y * BLOCK_SIZE + k] + shared[span][k * BLOCK_SIZE + threadIdx.x];
                tmp = min(tmp, center[threadIdx.y * BLOCK_SIZE + k] + shared[span][k * BLOCK_SIZE + threadIdx.x]);
                // __syncthreads();
                // shared[span][local_pos] = min(shared[span][local_pos], tmp);
                // __syncthreads();
            }
            // __syncthreads();
            // if (i < n && col_j < n)
                graph[i * n + col_j] = tmp;

            col_j += BLOCK_SIZE;
        }

    } else { // column

        int row_i = blockIdx.x * SPAN * BLOCK_SIZE + threadIdx.y;
        for (int span = 0; span < SPAN; ++span) {
            if (row_i < n && j < n) {
                shared[span][local_pos] = graph[row_i * n + j];
            } else {
                shared[span][local_pos] = M;
            }  
            row_i += BLOCK_SIZE;
        }

        __syncthreads();

        if (j >= n) return;
        row_i = blockIdx.x * SPAN * BLOCK_SIZE + threadIdx.y;
        for (int span = 0; span < SPAN; ++span) {
            if (row_i >= n) return;

            // tmp = shared[span][local_pos];
            tmp = M;
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                // tmp = shared[span][threadIdx.y * BLOCK_SIZE + k] + center[k * BLOCK_SIZE + threadIdx.x];
                tmp = min(tmp, shared[span][threadIdx.y * BLOCK_SIZE + k] + center[k * BLOCK_SIZE + threadIdx.x]);
                // __syncthreads();
                // shared[span][local_pos] = min(shared[span][local_pos], tmp);
                // __syncthreads();
            }
            // __syncthreads();
            // if (j < n && row_i < n)
                graph[row_i * n + j] = tmp;

            row_i += BLOCK_SIZE;
        }
    }
}

__global__ void phase3(int n, int p, int *graph) {
    __shared__ int row_shared[FIXED_SPAN][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int col_shared[FIXED_SPAN][BLOCK_SIZE * BLOCK_SIZE];
    int tmp;

    int i = p * BLOCK_SIZE + threadIdx.y;
    int j = p * BLOCK_SIZE + threadIdx.x;
    int local_pos = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    int col_j = blockIdx.x * FIXED_SPAN * BLOCK_SIZE + threadIdx.x;
    // if (i >= n) {
    //     for (int span = 0; span < FIXED_SPAN; ++span) {
    //         row_shared[span][local_pos] = M;
    //         col_j += BLOCK_SIZE;
    //     }
    // } else {
        for (int span = 0; span < FIXED_SPAN; ++span) {
            // if (col_j < n) {
            if (i < n && col_j < n) {
                row_shared[span][local_pos] = graph[i * n + col_j];
            } else {
                row_shared[span][local_pos] = M;
            }
            col_j += BLOCK_SIZE;
        }
    // }

    int row_i = blockIdx.y * FIXED_SPAN * BLOCK_SIZE + threadIdx.y;
    // if (j >= n) {
    //     for (int span = 0; span < FIXED_SPAN; ++span) {
    //         col_shared[span][local_pos] = M;
    //         row_i += BLOCK_SIZE;
    //     }
    // } else {
        for (int span = 0; span < FIXED_SPAN; ++span) {
            // if (row_i < n) {
            if (j < n && row_i < n) {
                col_shared[span][local_pos] = graph[row_i * n + j];
            } else {
                col_shared[span][local_pos] = M;
            }
            row_i += BLOCK_SIZE;
        }
    // }

    __syncthreads();

    // if ((blockIdx.y + 1) * FIXED_SPAN * BLOCK_SIZE + threadIdx.y < n && (blockIdx.x + 1) * FIXED_SPAN * BLOCK_SIZE + threadIdx.x < n) {
    if (row_i - BLOCK_SIZE < n && col_j - BLOCK_SIZE < n) {

    row_i = blockIdx.y * FIXED_SPAN * BLOCK_SIZE + threadIdx.y;
    for (int span1 = 0; span1 < FIXED_SPAN; ++span1) {

        col_j = blockIdx.x * FIXED_SPAN * BLOCK_SIZE + threadIdx.x;
        for (int span2 = 0; span2 < FIXED_SPAN; ++span2) {

            tmp = graph[row_i * n + col_j];
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                tmp = min(tmp, col_shared[span1][threadIdx.y * BLOCK_SIZE + k] 
                             + row_shared[span2][k * BLOCK_SIZE + threadIdx.x]);
            }
            graph[row_i * n + col_j] = tmp;

            col_j += BLOCK_SIZE;
        }

        row_i += BLOCK_SIZE;
    }

    } else {

    row_i = blockIdx.y * FIXED_SPAN * BLOCK_SIZE + threadIdx.y;
    for (int span1 = 0; span1 < FIXED_SPAN; ++span1) {
        // if (row_i == i) continue;
        if (row_i >= n) return;

        col_j = blockIdx.x * FIXED_SPAN * BLOCK_SIZE + threadIdx.x;
        for (int span2 = 0; span2 < FIXED_SPAN; ++span2) {
            // if (col_j == j) continue;
            if (col_j >= n) break;

            tmp = graph[row_i * n + col_j];
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                tmp = min(tmp, col_shared[span1][threadIdx.y * BLOCK_SIZE + k] 
                             + row_shared[span2][k * BLOCK_SIZE + threadIdx.x]);
            }
            // if (row_i < n && col_j < n)
                graph[row_i * n + col_j] = tmp;

            col_j += BLOCK_SIZE;
        }

        row_i += BLOCK_SIZE;
    }

    }
}

}

void apsp(int n, /* device */ int *graph) {
    // static_assert((SPAN + 1) * BLOCK_SIZE * BLOCK_SIZE * sizeof(int) <= 48 * 1024, 
    //               "SPAN chosen too large that overflows shared memory");
    // static_assert((6 * 1024) % (BLOCK_SIZE * BLOCK_SIZE) == 0, 
    //               "BLOCK_SIZE chosen poorly that shared memory cannot be used to the FULL extent");

    dim3 block;
    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE;

    dim3 grid2, grid3;
    grid2.x = (n - 1) / (SPAN * BLOCK_SIZE) + 1;
    grid2.y = 2;

    grid3.x = (n - 1) / (FIXED_SPAN * BLOCK_SIZE) + 1;
    grid3.y = (n - 1) / (FIXED_SPAN * BLOCK_SIZE) + 1;

    for (int p = 0; p < (n - 1) / BLOCK_SIZE + 1; ++p) {
        phase1<<<1, block>>>(n, p, graph);
        phase2<<<grid2, block>>>(n, p, graph);
        phase3<<<grid3, block>>>(n, p, graph);
    }

    // double dur1 = 0, dur2 = 0, dur3 = 0; 

    // for (int p = 0; p < (n - 1) / BLOCK_SIZE + 1; ++p) {
    //     auto begin = ch::high_resolution_clock::now();
    //     phase1<<<1, block>>>(n, p, graph);
    //     checkCudaErrors(cudaDeviceSynchronize());
    //     auto end = ch::high_resolution_clock::now();
    //     dur1 += ch::duration_cast<ch::duration<double>>(end - begin).count() * 1000;

    //     begin = ch::high_resolution_clock::now();
    //     phase2<<<grid2, block>>>(n, p, graph);
    //     checkCudaErrors(cudaDeviceSynchronize());
    //     end = ch::high_resolution_clock::now();
    //     dur2 += ch::duration_cast<ch::duration<double>>(end - begin).count() * 1000;

    //     begin = ch::high_resolution_clock::now();
    //     phase3<<<grid3, block>>>(n, p, graph);
    //     checkCudaErrors(cudaDeviceSynchronize());
    //     end = ch::high_resolution_clock::now();
    //     dur3 += ch::duration_cast<ch::duration<double>>(end - begin).count() * 1000;
    // }

    // printf("phase1: %f ms\nphase2: %f ms\nphase3: %f ms\n\n", dur1, dur2, dur3);
}

