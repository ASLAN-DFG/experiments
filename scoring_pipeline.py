from args import parse_args
from data.data_utils.data_reader import AnswerReader
from scoring.train_bert import train

if __name__ == '__main__':
    args = parse_args()

    # Read data
    train_fp = 'data/' + args.dataset + '/' + args.train
    test_fp = 'data/' + args.dataset + '/' + args.test

    # Preprocess data
    reader_train = AnswerReader(train_fp, args.dataset)
    df_train = reader_train.to_dataframe()
    reader_test = AnswerReader(test_fp, args.dataset)
    df_test = reader_test.to_dataframe()

    # Train (and currently Evaluation)
    train(args.experiment_name, df_train, df_test)

