#!/bin/bash

# assume the first arg is the exec file and the second arg is num of elements!
num_elements=$2

if [ $num_elements -le 2000 ]
then
    N=1
    n=1
elif [ $num_elements -le 8000 ]
then
    N=1
    n=4
elif [ $num_elements -le 20000 ]
then
    N=1
    n=8
elif [ $num_elements -le 50000 ]
then
    N=1
    n=10
elif [ $num_elements -le 200000 ]
then
    N=1
    n=16
elif [ $num_elements -le 500000 ]
then
    N=1
    n=20
elif [ $num_elements -le 800000 ]
then
    N=1
    n=25
elif [ $num_elements -le 1000000 ]
then
    N=1
    n=28
else
    N=2
    n=56
fi

srun -N $N -n $n --cpu-bind=cores $*
