#!/bin/bash

pids=$(ps -e | grep cpu_power_measu | awk '{ print $1 }' | sort | uniq)

for pid in $pids
do
    kill $pid
done

echo "All CPU measurement processes have been killed"