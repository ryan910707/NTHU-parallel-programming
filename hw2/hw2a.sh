#!/bin/bash

for c in {1..12}; do
    echo "$c thread"
    srun -pjudge -c$c time ./hw2a ./exp01.png 174170376 -0.7894722222222222 -0.7825277777777778 0.145046875 0.148953125 2549 1439
done
