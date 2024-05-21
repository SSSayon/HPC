#!/bin/bash

unroll_values=(1 2 4 8 16)

for unroll in ${unroll_values[@]}; do
    make clean
    make UNROLL_N=$unroll
    ./main 
done
