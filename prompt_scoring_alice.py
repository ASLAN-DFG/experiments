from args import parse_args
import pandas as pd
from data.data_utils.data_reader import AnswerReader
from scoring.train_bert import train


def clean_id(id_string):
    for c in '[\']':
        id_string = id_string.replace(c, "")
    if ',' in id_string:
        return id_string.split(', ')
    return id_string


if __name__ == '__main__':
    args = parse_args()

    # Read data
    train_fp = 'data/' + args.dataset + '/' + args.train
    test_fp = 'data/' + args.dataset + '/' + args.test

    # Map prompts
    prompt_info = 'data/' + args.dataset + '/question_numbering.tsv'
    id_map = pd.read_csv(prompt_info, sep='\t')
    id_cols_in_map = []
    if '2way' in args.train:
        id_cols_in_map = ['our_id', '2way_train_ids', '2way_trial_ids']
    else:
        id_cols_in_map = ['our_id', '3way_train_ids', '3way_trial_ids']
    id_map = id_map[id_cols_in_map].copy()

    for index, row in id_map.iterrows():
        our_id = row['our_id']
        train_question_id = clean_id(row[1])
        trial_question_id = clean_id(row[2])
        print(f"ID: {our_id} | Train: {train_question_id} | Trial: {trial_question_id}")
        reader_train = AnswerReader(train_fp, args.dataset, args.experiment_setting, train_question_id)
        df_train = reader_train.to_dataframe()
        reader_test = AnswerReader(test_fp, args.dataset, args.experiment_setting, trial_question_id)
        df_test = reader_test.to_dataframe()

        train(args.experiment_name, df_train, df_test, our_id)
