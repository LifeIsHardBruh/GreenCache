#!/bin/bash
model="neuralmagic/Meta-Llama-3-70B-Instruct-quantized.w8a16"
cache_size=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# max index of slices we want to run
slices_total_number=1
lams=(0.5 1 1.5 2 2.5)

for lam in ${lams[@]}
do
    for size in ${cache_size[@]}
    do 
        for slice_index in $(seq 0 $slices_total_number)
        do
        if [ "$size" -eq 0 ]; then
            # Initialize LMCache with max_local_cache_size = 0
            sed -i 's/max_local_cache_size: [0-9]\+/max_local_cache_size: 0/' ~/LMCache/benchmarks/multi-round-qa/example.yaml
            echo "start initialize LMCache with cache"
            ./initialize_lmcache.sh $model > ~/LMCache/benchmarks/multi-round-qa/tmp_output/lmc_output.txt 2>&1 &
            sleep 180

            # Start CPU power measurement and save the pid
            echo "start CPU power measurement"
            # $1: task_name, $2: slice_index, $3: size, $4: lam
            ./cpu_power_measurement.sh "lmc" $slice_index $size $lam &
            cpu_power_measurement_pid=$!

            # $1: model, $2: slice_index, $3: size, $4: lam
            echo "start 70BMulti_lmc"
            ./70BMulti_lmc_cache0.sh $model $slice_index $size $lam

            # Kill LMCache and delete cache and kill CPU power measurement
            echo "kill LMCache and delete cache"
            ./kill.sh
            kill -SIGTERM $cpu_power_measurement_pid
            ./del_cache.sh
            sleep 90
        else 
            # Initialize LMCache with max_local_cache_size = 40
            sed -i 's/max_local_cache_size: [0-9]\+/max_local_cache_size: 40/' ~/LMCache/benchmarks/multi-round-qa/example.yaml
            echo "start initialize LMCache with cache"
            ./initialize_lmcache.sh $model > ~/LMCache/benchmarks/multi-round-qa/tmp_output/lmc_output.txt 2>&1 &
            sleep 180

            # Start CPU power measurement and save the pid
            echo "start CPU power measurement"
            # $1: task_name, $2: slice_index, $3: size, $4: lam
            ./cpu_power_measurement.sh "lmc" $slice_index $size $lam &
            cpu_power_measurement_pid=$!

            # $1: model, $2: slice_index, $3: size, $4: lam
            echo "start 70BMulti_lmc"
            ./70BMulti_lmc.sh $model $slice_index $size $lam

            # Kill LMCache and delete cache and kill CPU power measurement
            echo "kill LMCache and delete cache"
            ./kill.sh
            kill -SIGTERM $cpu_power_measurement_pid
            ./del_cache.sh
            sleep 90
        fi
        done
    done
done
echo "Finish test"
