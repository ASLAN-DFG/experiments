#!/bin/bash

start_time= date '+%F %X'
echo $start_time
python3.8 prompt_scoring_alice.py --dataset alice --experiment_name prompt_instance_scoring_alice_2way_only_answer --train 2way/ALICE_LP_train_2way.json --test 2way/ALICE_LP_trial_2way.json
python3.8 prompt_scoring_alice.py --dataset alice --experiment_name prompt_instance_scoring_alice_2way_plus_question --experiment_setting with_question --train 2way/ALICE_LP_train_2way.json --test 2way/ALICE_LP_trial_2way.json
python3.8 prompt_scoring_alice.py --dataset alice --experiment_name prompt_instance_scoring_alice_2way_plus_rubric --experiment_setting with_rubric --train 2way/ALICE_LP_train_2way.json --test 2way/ALICE_LP_trial_2way.json
python3.8 prompt_scoring_alice.py --dataset alice --experiment_name prompt_instance_scoring_alice_2way_plus_both --experiment_setting with_both --train 2way/ALICE_LP_train_2way.json --test 2way/ALICE_LP_trial_2way.json
end_time= date '+%F %X'
echo $end_time