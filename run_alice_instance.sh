#!/bin/bash

start_time= date '+%F %X'
echo $start_time
python3.8 scoring_pipeline.py --dataset alice --experiment_name instance_based_scoring_alice_2way_question --experiment_setting with_question --train 2way/ALICE_LP_train_2way.json --test 2way/ALICE_LP_trial_2way.json
python3.8 scoring_pipeline.py --dataset alice --experiment_name instance_based_scoring_alice_2way_rubric --experiment_setting with_rubric --train 2way/ALICE_LP_train_2way.json --test 2way/ALICE_LP_trial_2way.json
python3.8 scoring_pipeline.py --dataset alice --experiment_name instance_based_scoring_alice_2way_both --experiment_setting with_both --train 2way/ALICE_LP_train_2way.json --test 2way/ALICE_LP_trial_2way.json
python3.8 scoring_pipeline.py --dataset alice --experiment_name instance_based_scoring_alice_3way_question --experiment_setting with_question --train 3way/ALICE_LP_train_3way.json --test 3way/ALICE_LP_trial_3way.json
python3.8 scoring_pipeline.py --dataset alice --experiment_name instance_based_scoring_alice_3way_rubric --experiment_setting with_rubric --train 3way/ALICE_LP_train_3way.json --test 3way/ALICE_LP_trial_3way.json
python3.8 scoring_pipeline.py --dataset alice --experiment_name instance_based_scoring_alice_3way_both --experiment_setting with_both --train 3way/ALICE_LP_train_3way.json --test 3way/ALICE_LP_trial_3way.json
end_time= date '+%F %X'
echo $end_time