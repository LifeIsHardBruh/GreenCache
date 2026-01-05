#!/bin/bash

# $1: task_name, $2: slice_index, $3: size, $4: lam
task_name=$1
slice_index=$2
size=$3
lam=$4

#input your sudo pwd here since the cpu measurement can only be done in sudo mode
pwd='xxxxxxx'

trap 'echo "Received SIGTERM, cleaning up cpu_power_measurement..."; exit 0' SIGTERM

while true; do
    current_time=$(date "+%s")
    power=$(echo $pwd | sudo -S ./ryzen | tail -n 1)
    output="Time: ${current_time}, ${power}"
    echo "$output" >> "../70BMulti_output/70BMulti_${task_name}_CPU_power_${slice_index}_${size}_${lam}.txt"
    sleep 1
done