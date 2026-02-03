import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)

    return parser.parse_args()