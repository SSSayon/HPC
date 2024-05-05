#!/bin/bash

make

for stride in 1 2 4 8
do
    sed -i "s/#define STRIDE .*/#define STRIDE $stride/" test_gmem.cu
    make
    echo "Running test_gmem.cu with STRIDE=$stride"
    srun --exclusive --gres=gpu:1 ./test_gmem >> results_gmem.txt
done

for bitwidth in 2 4 8
do
    for stride in 1 2 4 8 16 32
    do
        sed -i "s/#define BITWIDTH .*/#define BITWIDTH $bitwidth/" test_smem.cu
        sed -i "s/#define STRIDE .*/#define STRIDE $stride/" test_smem.cu
        make
        echo "Running test_smem.cu with BITWIDTH=$bitwidth, STRIDE=$stride"
        srun --exclusive --gres=gpu:1 ./test_smem >> results_smem2.txt
    done
done
