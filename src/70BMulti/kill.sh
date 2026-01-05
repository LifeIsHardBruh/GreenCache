#!/bin/bash

pids=$(nvidia-smi | grep 'python' | awk '{ print $5 }' | sort | uniq)

for pid in $pids
do
    kill $pid
done

echo "All GPU python processes have been killed."