#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"


inline int get_block_len(int block_size, int n, int rank) {
    return std::max(std::min(block_size, n - block_size * rank), 0);
}

inline int send_prev(int rank) {
    return (rank == 0) ? MPI_PROC_NULL : rank - 1;
}

inline int send_next(int rank, int nprocs) {
    return (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;
}

void merge_given_num(const float *data1, int n1, const float *data2, int n2, // data2 must belong to current rank
                     float *merged, int n,                                   // actually n2 == n
                     int flag, int *local_flag) {                            // flag: 0-smallest, 1-largest
    
    if (flag) { // largest n-th ones
        int i = n1 - 1, j = n2 - 1, k = n - 1;
        while (i >= 0 && j >= 0 && k >= 0) {
            if (data1[i] > data2[j]) {
                *local_flag = 1;
                merged[k--] = data1[i--]; 
            } else {
                merged[k--] = data2[j--];
            }
        }
        while (i >= 0 && k >= 0) {
            merged[k--] = data1[i--];
        }
        while (j >= 0 && k >= 0) {
            merged[k--] = data2[j--];
        }
    } else { // smallest n-th ones
        int i = 0, j = 0, k = 0;
        while (i < n1 && j < n2 && k < n) {
            if (data1[i] < data2[j]) {
                *local_flag = 1;
                merged[k++] = data1[i++]; 
            } else {
                merged[k++] = data2[j++];
            }
        }
        while (i < n1 && k < n) {
            merged[k++] = data1[i++];
        }
        while (j < n2 && k < n) {
            merged[k++] = data2[j++];
        }
    }

    return;
}

void Worker::sort() {

    std::sort(data, data + block_len);
    if (nprocs == 1) return;

    int block_size = ceiling(n, nprocs); 
    // use `get_block_len()` to DIRECTLY calc nbr's block_len, rather than communicating
    int prev_or_next_block_len = 0;

    // IMPORTANT!!! introduce a new comm of valid processes
    // NOTE: THIS STEP MUST BE AHEAD OF THE `return` SENTENCE OF INVALID PROCESSES!!!
    int valid_nproc = (n + block_size - 1)/ block_size;
    int *valid_ranks = new int[valid_nproc];
    for (int i = 0; i < valid_nproc; ++i) {
        valid_ranks[i] = i;
    }
    MPI_Group world_group, valid_group;
    MPI_Comm valid_comm; // only used in MPI_BARRIER. NOTE: CANNOT USE IN SEND/RECV, AS THE CORRESPONDING RANK WILL BE REARRANGED!!!
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, valid_nproc, valid_ranks, &valid_group);
    MPI_Comm_create(MPI_COMM_WORLD, valid_group, &valid_comm);

    // SO THAT THE PROCESSES BELOW ARE ALL VALID
    if (out_of_range) return;
    
    float *recv_buf = new float[block_size];
    float *merged_buf = new float[block_len]; 

    MPI_Request req[2];

    int needs_merge = 0;
    float prev_last_or_next_first_element = 0.f;

    int global_flag = 1;
    int local_flag = 1;
    while (global_flag) {

        local_flag = 0;
        global_flag = 0;

        // Even phase ----------------------------------------------------------
        prev_or_next_block_len = 0;
        needs_merge = 0;
        prev_last_or_next_first_element = 0.f;

        // check whether needs to merge
        if (rank % 2 == 0 && rank != valid_nproc - 1) {
            MPI_Isend(data + block_len - 1, 1, MPI_FLOAT, rank + 1, rank, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(&prev_last_or_next_first_element, 1, MPI_FLOAT, rank + 1, rank + 1, MPI_COMM_WORLD, &req[1]);
        } else if (rank % 2 == 1) {
            MPI_Isend(data, 1, MPI_FLOAT, rank - 1, rank, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(&prev_last_or_next_first_element, 1, MPI_FLOAT, rank - 1, rank - 1, MPI_COMM_WORLD, &req[1]);
        }

        if (rank % 2 == 0 && rank != valid_nproc - 1) {
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
            if (data[block_len - 1] > prev_last_or_next_first_element) {
                needs_merge = 1;
            }
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        } else if (rank % 2 == 1) {
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
            if (data[0] < prev_last_or_next_first_element) {
                needs_merge = 1;
            }
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        }

        // merge
        if (needs_merge) {
            if (rank % 2 == 0) {
                prev_or_next_block_len = get_block_len(block_size, n, rank + 1);
                MPI_Isend(data, block_len, MPI_FLOAT, rank + 1, rank, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(recv_buf, prev_or_next_block_len, MPI_FLOAT, 
                        rank + 1, rank + 1, MPI_COMM_WORLD, &req[1]);
            } else {
                prev_or_next_block_len = get_block_len(block_size, n, rank - 1);
                MPI_Isend(data, block_len, MPI_FLOAT, rank - 1, rank, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(recv_buf, prev_or_next_block_len, MPI_FLOAT, 
                          rank - 1, rank - 1, MPI_COMM_WORLD, &req[1]);
            }
        }

        if (needs_merge) {
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
            if (rank % 2 == 0) {
                merge_given_num(recv_buf, prev_or_next_block_len, data, block_len, 
                                merged_buf, block_len, 0, &local_flag);
            } else {
                merge_given_num(recv_buf, prev_or_next_block_len, data, block_len, 
                                merged_buf, block_len, 1, &local_flag);
            }
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            std::copy(merged_buf, merged_buf + block_len, data);
        }

        MPI_Barrier(valid_comm);


        // Odd phase ----------------------------------------------------------
        prev_or_next_block_len = 0;
        needs_merge = 0;
        prev_last_or_next_first_element = 0.f;

        // check whether needs to merge
        if (rank % 2 == 1 && rank != valid_nproc - 1) {
            MPI_Isend(data + block_len - 1, 1, MPI_FLOAT, rank + 1, rank, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(&prev_last_or_next_first_element, 1, MPI_FLOAT, rank + 1, rank + 1, MPI_COMM_WORLD, &req[1]);
        } else if (rank % 2 == 0 && rank != 0) {
            MPI_Isend(data, 1, MPI_FLOAT, rank - 1, rank, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(&prev_last_or_next_first_element, 1, MPI_FLOAT, rank - 1, rank - 1, MPI_COMM_WORLD, &req[1]);
        }

        if (rank % 2 == 1 && rank != valid_nproc - 1) {
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
            if (data[block_len - 1] > prev_last_or_next_first_element) {
                needs_merge = 1;
            }
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        } else if (rank % 2 == 0 && rank != 0) {
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
            if (data[0] < prev_last_or_next_first_element) {
                needs_merge = 1;
            }
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        }

        // merge
        if (needs_merge) {
            if (rank % 2 == 1) {
                prev_or_next_block_len = get_block_len(block_size, n, rank + 1);
                MPI_Isend(data, block_len, MPI_FLOAT, rank + 1, rank, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(recv_buf, prev_or_next_block_len, MPI_FLOAT, 
                        rank + 1, rank + 1, MPI_COMM_WORLD, &req[1]);
            } else {
                prev_or_next_block_len = get_block_len(block_size, n, rank - 1);
                MPI_Isend(data, block_len, MPI_FLOAT, rank - 1, rank, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(recv_buf, prev_or_next_block_len, MPI_FLOAT, 
                          rank - 1, rank - 1, MPI_COMM_WORLD, &req[1]);
            }
        }

        if (needs_merge) {
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
            if (rank % 2 == 1) {
                merge_given_num(recv_buf, prev_or_next_block_len, data, block_len, 
                                merged_buf, block_len, 0, &local_flag);
            } else {
                merge_given_num(recv_buf, prev_or_next_block_len, data, block_len, 
                                merged_buf, block_len, 1, &local_flag);
            }
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            std::copy(merged_buf, merged_buf + block_len, data);
        }

        MPI_Barrier(valid_comm);


        // Allreduce global_flag ----------------------------------------------------------
        if (rank == valid_nproc - 1) {
            MPI_Send(&local_flag, 1, MPI_INT, send_prev(rank), rank, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&global_flag, 1, MPI_INT, rank + 1, rank + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_flag |= local_flag;
            MPI_Send(&global_flag, 1, MPI_INT, send_prev(rank), rank, MPI_COMM_WORLD);
        }
        MPI_Barrier(valid_comm);
        if (rank == 0) {
            MPI_Send(&global_flag, 1, MPI_INT, send_next(rank, valid_nproc), rank, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&global_flag, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&global_flag, 1, MPI_INT, send_next(rank, valid_nproc), rank, MPI_COMM_WORLD);
        }
        MPI_Barrier(valid_comm);
    }

    delete[] recv_buf;
    delete[] merged_buf;
}
