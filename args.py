import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)

    # Clustering Parameters
    parser.add_argument("--clustering_method", type=str, default='k_means', required=False)
    parser.add_argument("--n_clusters", type=int, default=10, required=False)

    return parser.parse_args()