#!/bin/bash

# Array of TRUNCATED_STEP values to test
steps=(32 64 128 256 512)

datasets=("arxiv" "collab" "citation" "ddi" "protein" "ppa" "reddit.dgl" "products" "youtube" "amazon_cogdl" "yelp" "wikikg2" "am")

for dataset in "${datasets[@]}"; do
output_file="gen/$dataset.txt"

# Clear the output file
> $output_file

for step in "${steps[@]}"; do
    # Modify TRUNCATED_STEP in spmm_opt.cu
    sed -i "s/#define TRUNCATED_STEP [0-9]*/#define TRUNCATED_STEP $step/" ../src/spmm_opt.cu
    
    # Compile the project
    cmake .. && make -j4
    
    # Run the tests and capture the output (both stdout and stderr)
    test_output=$(srun -N 1 --gres=gpu:1 ./test/unit_tests --dataset $dataset --len 32 --datadir ~/PA3/data/ 2>&1)
    
    # Write the TRUNCATED_STEP value and the output to the file
    echo "TRUNCATED_STEP=$step" >> $output_file
    echo "$test_output" >> $output_file
    echo -e "\n" >> $output_file
done

done
