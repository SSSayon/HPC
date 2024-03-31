#include <chrono>
#include <iostream>
#include <mpi.h>
#include <time.h>
#include <cstring>
#include <cmath>
#include <algorithm>

#define EPS 1e-8

namespace ch = std::chrono;

inline int get_block_num(int rank, int step, int module, int avg_size, bool flag) {
    if (flag) return ((rank - step) % module + module) % module;
    else return ((rank + 1 - step) % module + module) % module;
}

inline int sendto(int rank, int comm_sz) {
    if (rank == comm_sz - 1) return 0;
    return rank + 1;
}

inline int recvfrom(int rank, int comm_sz) {
    if (rank == 0) return comm_sz - 1;
    return rank - 1; 
}

inline int get_block_size(int block_num, int P, int avg_size, int delta) {
    if (block_num == P - 1) return avg_size + delta;
    return avg_size;
}

void Ring_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank) {
    float* sendbuf_float = static_cast<float*>(sendbuf);
    float* recvbuf_float = static_cast<float*>(recvbuf);

    // partition
    int P = comm_sz;
    int avg_size = n / comm_sz;
    int delta =  n - avg_size * comm_sz;
    // printf("RANK %d SIZE %d\n", my_rank, block_size);

    // reduce-scatter
    for (int step = 0; step < P-1; ++step) {

        MPI_Request req[2];

        int recv_from = recvfrom(my_rank, comm_sz);
        int recv_block_num = get_block_num(recv_from, step, P, avg_size, 1);
        int recv_block_size = get_block_size(recv_block_num, P, avg_size, delta);
        MPI_Irecv(recvbuf_float + recv_block_num * avg_size, 
                  recv_block_size, MPI_FLOAT, recv_from, recv_from, comm, &req[0]);

        int send_block_num = get_block_num(my_rank, step, P, avg_size, 1);
        MPI_Isend(sendbuf_float + send_block_num * avg_size, 
                  get_block_size(send_block_num, P, avg_size, delta), MPI_FLOAT, sendto(my_rank, comm_sz), my_rank, comm, &req[1]);

        // MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);

        for (int i = recv_block_num * avg_size; i < recv_block_num * avg_size + recv_block_size; ++i) {
            sendbuf_float[i] += recvbuf_float[i];
        }

        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        // MPI_Barrier(MPI_COMM_WORLD);
    }

    memcpy(recvbuf_float, sendbuf_float, n * sizeof(float));

    // allgather
    for (int step = 0; step < P-1; ++step) {
        int recv_from = recvfrom(my_rank, comm_sz);
        int recv_block_num = get_block_num(recv_from, step, P, avg_size, 0);
        int recv_block_size = get_block_size(recv_block_num, P, avg_size, delta);

        int send_block_num = get_block_num(my_rank, step, P, avg_size, 0);

        MPI_Sendrecv(recvbuf_float + send_block_num * avg_size, get_block_size(send_block_num, P, avg_size, delta), MPI_FLOAT, 
                     sendto(my_rank, comm_sz), my_rank,
                     recvbuf_float + recv_block_num * avg_size, recv_block_size, MPI_FLOAT, recv_from, recv_from,
                     comm, MPI_STATUS_IGNORE);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // MPI_Barrier(MPI_COMM_WORLD);
}

// reduce + bcast
void Naive_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    MPI_Reduce(sendbuf, recvbuf, n, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Bcast(recvbuf, n, MPI_FLOAT, 0, comm);
}

int main(int argc, char *argv[])
{
    int ITER = atoi(argv[1]);
    int n = atoi(argv[2]);
    float* mpi_sendbuf = new float[n];
    float* mpi_recvbuf = new float[n];
    float* naive_sendbuf = new float[n];
    float* naive_recvbuf = new float[n];
    float* ring_sendbuf = new float[n];
    float* ring_recvbuf = new float[n];

    MPI_Init(nullptr, nullptr);
    int comm_sz;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    srand(time(NULL) + my_rank);
    for (int i = 0; i < n; ++i)
        mpi_sendbuf[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    memcpy(naive_sendbuf, mpi_sendbuf, n * sizeof(float));
    memcpy(ring_sendbuf, mpi_sendbuf, n * sizeof(float));

    //warmup and check
    MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (abs(mpi_recvbuf[i] - ring_recvbuf[i]) > EPS)
        {
            correct = false;
            break;
        }

    if (correct)
    {
        auto beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto end = ch::high_resolution_clock::now();
        double mpi_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double naive_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double ring_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms
        
        if (my_rank == 0)
        {
            std::cout << "Correct." << std::endl;
            std::cout << "MPI_Allreduce:   " << mpi_dur << " ms." << std::endl;
            std::cout << "Naive_Allreduce: " << naive_dur << " ms." << std::endl;
            std::cout << "Ring_Allreduce:  " << ring_dur << " ms." << std::endl;
        }
    }
    else
        if (my_rank == 0)
            std::cout << "Wrong!" << std::endl;

    delete[] mpi_sendbuf;
    delete[] mpi_recvbuf;
    delete[] naive_sendbuf;
    delete[] naive_recvbuf;
    delete[] ring_sendbuf;
    delete[] ring_recvbuf;
    MPI_Finalize();
    return 0;
}
