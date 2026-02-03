#!/bin/bash

start_time= date '+%F %X'
echo $start_time
for K in 1 2 3 4 5 6 7 8 9 10
do
    python3.8 scoring_pipeline.py --dataset ASAP --experiment_name instance_based_scoring_asap --train $K'_GOLD_train.csv' --test $K'_GOLD_test.csv'
done
end_time= date '+%F %X'
echo $end_time