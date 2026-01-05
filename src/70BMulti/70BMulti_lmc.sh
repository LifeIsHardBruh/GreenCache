#!/bin/bash
# $1: model, $2: slice_index, $3: size, $4: lam

model=$1
slice_index=$2
size=$3
lam=$4

python 70BMulti_script.py \
--chat-history "../../chat_history_full_dataset.pickle" \
--size $size \
--slices 2 \
--request-num 500 \
--begining-index 200000 \
--output-dir "../70BMulti_intermediate/"

python ./multi-round-qa_70BMulti.py --lam $lam \
--duration 9999 \
--model $model \
--base-url http://localhost:8000/v1 \
--output "../70BMulti_output/70BMulti_lmc_${slice_index}_${size}_${lam}.csv" \
--pickle-file-path "../70BMulti_intermediate/cache_chat_histories_${slice_index}.pickle" \
--cache-list "../70BMulti_intermediate/cache_list_${slice_index}.txt" \
--integration \
--realTime-pickle-file-path "../70BMulti_intermediate/realTime_chat_histories_${slice_index}.pickle" \
--power-usage-output "../70BMulti_output/70BMulti_lmc_power_${slice_index}_${size}_${lam}.csv"

sleep 10

python ./multi-round-qa_70BMulti.py --lam $lam \
--duration 9999 \
--model $model \
--base-url http://localhost:8000/v1 \
--output "../70BMulti_output/70BMulti_lmc_${slice_index}_${size}_${lam}.csv" \
--pickle-file-path "../70BMulti_intermediate/realTime_chat_histories_${slice_index}.pickle" \
--power-usage-output "../70BMulti_output/70BMulti_lmc_power_${slice_index}_${size}_${lam}.csv"

sleep 10
