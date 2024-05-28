#include "../include/spmm_opt.h"
#include <vector>

/*
    variables we can access:
    
    int *d_ptr = NULL;
    int *d_idx = NULL;
    float *d_val = NULL;

    int feat_in = 0;

    int num_v = 0;
    int num_e = 0;

    dim3 grid;
    dim3 block;
*/

#define WARP_SIZE 32
#define K_256_STEP 2 

// __global__ void spmm_kernel_K_32(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
// {
//     // vin == B, vout == C, num_v == M, INFEATURE == K

//     __shared__ int idx_shared[WARP_SIZE];
//     __shared__ float val_shared[WARP_SIZE];

//     int row = blockIdx.x;
//     int col = threadIdx.x;

//     if (row >= num_v) return;

//     float res = 0.f;
//     int begin = ptr[row], end = ptr[row + 1];
//     for (int part = begin; part < end; part += WARP_SIZE) { // divide the row-th row of A into several parts, 
//                                                             // each contains <= WARP_SIZE nonzero elements
//         int ptr_id = part + col;
//         if (ptr_id < end) {
//             idx_shared[col] = idx[ptr_id];
//             val_shared[col] = val[ptr_id];
//         }

//         __syncthreads(); // REMEMBER TO DO THIS after load something to shared memory

//         for (int i = 0; i < min(WARP_SIZE, end - part); ++i) {
//             res += val_shared[i] * vin[idx_shared[i] * INFEATURE + col];
//         }
//     }
    
//     vout[row * INFEATURE + col] = res;
// }

__global__ void spmm_kernel_K_32_with_truncated_line(int *idx, float *val, float *vin, float *vout, int INFEATURE,
                                                     TRUNCATED_LINE *truncated_lines, int num_truncated_lines)
{
    __shared__ int idx_shared[WARP_SIZE];
    __shared__ float val_shared[WARP_SIZE];

    int col = threadIdx.x;
    int thread_id = blockIdx.x;

    if (thread_id >= num_truncated_lines) return;

    TRUNCATED_LINE truncated_line = truncated_lines[thread_id];
    int row = truncated_line.row;
    int begin = truncated_line.begin;
    int end = truncated_line.end;

    float res = 0.f;
    for (int part = begin; part < end; part += WARP_SIZE) {
        int ptr_id = part + col;
        if (ptr_id < end) {
            idx_shared[col] = idx[ptr_id];
            val_shared[col] = val[ptr_id];
        }

        __syncthreads();

        for (int i = 0; i < min(WARP_SIZE, end - part); ++i) {
            res += val_shared[i] * vin[idx_shared[i] * INFEATURE + col];
        }
    }

    atomicAdd(&vout[row * INFEATURE + col], res); // NEED TO USE atom operation there!
}

__global__ void spmm_kernel_K_32_with_combined_line(int *idx, float *val, float *vin, float *vout, int INFEATURE,
                                                    COMBINED_LINE *combined_lines, int num_combined_lines)
{
    int col = threadIdx.x;
    int thread_id = blockIdx.x;

    if (thread_id >= num_combined_lines) return;

    COMBINED_LINE combined_line = combined_lines[thread_id];

    for (int i = 0; i < combined_size; ++i) {
        
        __shared__ int idx_shared[max_element_per_line];
        __shared__ float val_shared[max_element_per_line];

        int row = combined_line.rows[i];
        int begin = combined_line.begins[i];
        int end = combined_line.ends[i];
        
        int ptr_id = begin + col;
        if (ptr_id < end) {
            idx_shared[col] = idx[ptr_id];
            val_shared[col] = val[ptr_id];
        }

        __syncthreads();

        float res = 0.f;
        for (int j = 0; j < min(end - begin, max_element_per_line); ++j) {
            res += val_shared[j] * vin[idx_shared[j] * INFEATURE + col];
        }

        atomicAdd(&vout[row * INFEATURE + col], res);

        __syncthreads();
    }
}

// __global__ void spmm_kernel_K_256(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
// {
//     __shared__ int idx_shared[WARP_SIZE];
//     __shared__ float val_shared[WARP_SIZE];

//     int row = blockIdx.x;
//     int col = blockIdx.y * (WARP_SIZE * K_256_STEP) + threadIdx.x;

//     if (row >= num_v) return;

//     float res = 0.f, res2 = 0.f;
//     int begin = ptr[row], end = ptr[row + 1];
//     for (int part = begin; part < end; part += WARP_SIZE) {
//         int ptr_id = part + threadIdx.x;
//         if (ptr_id < end) {
//             idx_shared[threadIdx.x] = idx[ptr_id];
//             val_shared[threadIdx.x] = val[ptr_id];
//         }

//         __syncthreads();

//         for (int i = 0; i < min(WARP_SIZE, end - part); ++i) {
//             res += val_shared[i] * vin[idx_shared[i] * INFEATURE + col];
//             res2 += val_shared[i] * vin[idx_shared[i] * INFEATURE + col + (K_256_STEP - 1) * WARP_SIZE];
//         }
//     }
    
//     vout[row * INFEATURE + col] = res;
//     vout[row * INFEATURE + col + (K_256_STEP - 1) * WARP_SIZE] = res2;
// }

__global__ void spmm_kernel_K_256_with_truncated_line(int *idx, float *val, float *vin, float *vout, int INFEATURE,
                                                      TRUNCATED_LINE *truncated_lines, int num_truncated_lines)
{
    __shared__ int idx_shared[WARP_SIZE];
    __shared__ float val_shared[WARP_SIZE];

    int col = blockIdx.y * (WARP_SIZE * K_256_STEP) + threadIdx.x;
    int thread_id = blockIdx.x;

    if (thread_id >= num_truncated_lines) return;

    TRUNCATED_LINE truncated_line = truncated_lines[thread_id];
    int row = truncated_line.row;
    int begin = truncated_line.begin;
    int end = truncated_line.end;

    float res = 0.f, res2 = 0.f;
    for (int part = begin; part < end; part += WARP_SIZE) {
        int ptr_id = part + threadIdx.x;
        if (ptr_id < end) {
            idx_shared[threadIdx.x] = idx[ptr_id];
            val_shared[threadIdx.x] = val[ptr_id];
        }

        __syncthreads();

        for (int i = 0; i < min(WARP_SIZE, end - part); ++i) {
            res += val_shared[i] * vin[idx_shared[i] * INFEATURE + col];
            res2 += val_shared[i] * vin[idx_shared[i] * INFEATURE + col + (K_256_STEP - 1) * WARP_SIZE];
        }
    }

    atomicAdd(&vout[row * INFEATURE + col], res);
    atomicAdd(&vout[row * INFEATURE + col + (K_256_STEP - 1) * WARP_SIZE], res2);
}

__global__ void spmm_kernel_K_256_with_combined_line(int *idx, float *val, float *vin, float *vout, int INFEATURE,
                                                     COMBINED_LINE *combined_lines, int num_combined_lines)
{
    int col = blockIdx.y * (WARP_SIZE * K_256_STEP) + threadIdx.x;
    int thread_id = blockIdx.x;

    if (thread_id >= num_combined_lines) return;

    COMBINED_LINE combined_line = combined_lines[thread_id];

    for (int i = 0; i < combined_size; ++i) {
        
        __shared__ int idx_shared[max_element_per_line];
        __shared__ float val_shared[max_element_per_line];

        int row = combined_line.rows[i];
        int begin = combined_line.begins[i];
        int end = combined_line.ends[i];
        
        int ptr_id = begin + threadIdx.x;
        if (ptr_id < end) {
            idx_shared[threadIdx.x] = idx[ptr_id];
            val_shared[threadIdx.x] = val[ptr_id];
        }

        __syncthreads();

        float res = 0.f, res2 = 0.f;
        for (int j = 0; j < min(end - begin, max_element_per_line); ++j) {
            res += val_shared[j] * vin[idx_shared[j] * INFEATURE + col];
            res2 += val_shared[j] * vin[idx_shared[j] * INFEATURE + col + (K_256_STEP - 1) * WARP_SIZE];
        }

        atomicAdd(&vout[row * INFEATURE + col], res);
        atomicAdd(&vout[row * INFEATURE + col + (K_256_STEP - 1) * WARP_SIZE], res2);

        __syncthreads();
    }
}

void SpMMOpt::preprocess_truncated_line(int truncated_step) 
{
    int *h_ptr = new int[num_v + 1]; // NOTE THIS +1
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<TRUNCATED_LINE> h_truncated_lines;
    for (int i = 0; i < num_v; ++i) {
        int begin = h_ptr[i], end = h_ptr[i+1];
        for (int j = begin; j < end; j += truncated_step) {
            TRUNCATED_LINE truncated_line = {i, j, min(j + truncated_step, end)};
            h_truncated_lines.push_back(truncated_line);
        }
    }
    num_truncated_lines = h_truncated_lines.size();
    checkCudaErrors(cudaMalloc(&d_truncated_lines, num_truncated_lines * sizeof(TRUNCATED_LINE)));
    checkCudaErrors(cudaMemcpy(d_truncated_lines, h_truncated_lines.data(), num_truncated_lines * sizeof(TRUNCATED_LINE), cudaMemcpyHostToDevice));

    grid.x = num_truncated_lines;
    if (feat_in == 256) grid.y = 8 / K_256_STEP;
    block.x = WARP_SIZE;

    delete[] h_ptr; // DO NOT FORGET THIS
}

void SpMMOpt::preprocess_trunc_combined_line(int truncated_step) 
{
    int *h_ptr = new int[num_v + 1];
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // truncated --------------------------------------
    std::vector<TRUNCATED_LINE> h_truncated_lines;
    for (int i = 0; i < num_v; ++i) {
        int begin = h_ptr[i], end = h_ptr[i+1];
        if (end - begin > max_element_per_line) {
            for (int j = begin; j < end; j += truncated_step) {
                TRUNCATED_LINE truncated_line = {i, j, min(j + truncated_step, end)};
                h_truncated_lines.push_back(truncated_line);
            }
        }
    }
    num_truncated_lines = h_truncated_lines.size();
    checkCudaErrors(cudaMalloc(&d_truncated_lines, num_truncated_lines * sizeof(TRUNCATED_LINE)));
    checkCudaErrors(cudaMemcpy(d_truncated_lines, h_truncated_lines.data(), num_truncated_lines * sizeof(TRUNCATED_LINE), cudaMemcpyHostToDevice));

    grid.x = num_truncated_lines;
    if (feat_in == 256) grid.y = 8 / K_256_STEP;
    block.x = WARP_SIZE;

    // combined --------------------------------------
    // std::vector<COMBINED_LINE> h_combined_lines;
    // int cnt = 0;
    // COMBINED_LINE combined_line;
    // for (int i = 0; i < num_v; ++i) {
    //     int begin = h_ptr[i], end = h_ptr[i+1];
    //     if ((end - begin > 0) && (end - begin <= max_element_per_line)) {
    //         combined_line.rows[cnt] = i;
    //         combined_line.begins[cnt] = begin;
    //         combined_line.ends[cnt] = end;

    //         cnt++;

    //         if (cnt == combined_size) {
    //             h_combined_lines.push_back(combined_line);
    //             cnt = 0;
    //             combined_line = COMBINED_LINE();
    //         }
    //     }
    // }
    // if (cnt != 0) {
    //     h_combined_lines.push_back(combined_line);
    // }

    // sort first. Seems not bring much benefit ...
    std::vector<COMBINED_LINE> h_combined_lines;
    std::vector<RowInfo> rows_info;

    for (int i = 0; i < num_v; ++i) {
        int begin = h_ptr[i], end = h_ptr[i + 1];
        if ((end - begin > 0) && (end - begin <= max_element_per_line)) {
            rows_info.push_back({i, begin, end});
        }
    }

    std::sort(rows_info.begin(), rows_info.end(), [](const RowInfo &a, const RowInfo &b) {
        return std::tie(a.begin, a.end) < std::tie(b.begin, b.end);
    });

    int cnt = 0;
    COMBINED_LINE combined_line;
    for (const RowInfo &row_info : rows_info) {
        combined_line.rows[cnt] = row_info.index;
        combined_line.begins[cnt] = row_info.begin;
        combined_line.ends[cnt] = row_info.end;

        cnt++;

        if (cnt == combined_size) {
            h_combined_lines.push_back(combined_line);
            cnt = 0;
            combined_line = COMBINED_LINE();
        }
    }
    if (cnt != 0) {
        h_combined_lines.push_back(combined_line);
    }

    num_combined_lines = h_combined_lines.size();
    checkCudaErrors(cudaMalloc(&d_combined_lines, num_combined_lines * sizeof(COMBINED_LINE)));
    checkCudaErrors(cudaMemcpy(d_combined_lines, h_combined_lines.data(), num_combined_lines * sizeof(COMBINED_LINE), cudaMemcpyHostToDevice));

    grid2.x = num_combined_lines;
    if (feat_in == 256) grid2.y = 8 / K_256_STEP;
    block2.x = WARP_SIZE;

    delete[] h_ptr;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // if (feat_in == 32) {
    //     grid.x = num_v;
    //     block.x = WARP_SIZE;
    // } else {
    //     grid.x = num_v;
    //     grid.y = 8; // 256 / 32
    //     block.x = WARP_SIZE;
    // }

    // truncated / combined lines

    int TRUNCATED_STEP = 128;
    switch (num_v) {
        case 169343:  TRUNCATED_STEP = 128;  break;
        case 235868:  TRUNCATED_STEP = 64;   break;
        case 2927963: TRUNCATED_STEP = 128;  break;
        case 4267:    TRUNCATED_STEP = 128;  break;
        case 132534:  TRUNCATED_STEP = 512;  break;
        case 576289:  TRUNCATED_STEP = 32;   break;
        case 232965:  TRUNCATED_STEP = 2048; break;
        case 2449029: TRUNCATED_STEP = 32;   break;
        case 1138499: TRUNCATED_STEP = 256;  break;
        case 1569960: TRUNCATED_STEP = 2048; break;
        case 716847:  TRUNCATED_STEP = 64;   break;
        case 2500604: TRUNCATED_STEP = 128;  break;
        case 881680:  TRUNCATED_STEP = 256;  break;
        default: break;
    }

    if (feat_in == 32) {
        // preprocess_truncated_line(TRUNCATED_STEP);
        preprocess_trunc_combined_line(TRUNCATED_STEP);  
    } else { // feat_in == 256
        // preprocess_truncated_line(TRUNCATED_STEP);
        preprocess_trunc_combined_line(TRUNCATED_STEP);
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    if (feat_in == 32) {
        // spmm_kernel_K_32<<<grid, block>>>
        //     (d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        spmm_kernel_K_32_with_truncated_line<<<grid, block>>>
            (d_idx, d_val, vin, vout, feat_in, d_truncated_lines, num_truncated_lines);
        spmm_kernel_K_32_with_combined_line<<<grid2, block2>>>
            (d_idx, d_val, vin, vout, feat_in, d_combined_lines, num_combined_lines);
    } else { // feat_in == 256
        // spmm_kernel_K_256<<<grid, block>>>
        //     (d_ptr, d_idx, d_val, vin, vout, num_v, feat_in); 
        spmm_kernel_K_256_with_truncated_line<<<grid, block>>>
            (d_idx, d_val, vin, vout, feat_in, d_truncated_lines, num_truncated_lines);
        spmm_kernel_K_256_with_combined_line<<<grid2, block2>>>
            (d_idx, d_val, vin, vout, feat_in, d_combined_lines, num_combined_lines);
    }
}
