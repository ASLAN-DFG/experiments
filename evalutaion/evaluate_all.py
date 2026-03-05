import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score

from clustering.clustering_evaluator import ClusterEvaluator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred", type=str, required=True, action='append')
    parser.add_argument("--embeddings", type=str, required=False, action='append')
    parser.add_argument("-o", "--output", type=str, default="out/all_scores.csv")

    return parser.parse_args()


def get_gold_column(df):
    for col in ["Score1"]:
        if col in df.columns:
            return col
    return None


def get_pred_columns(df):
    cols = []
    for col in ["predicted_score", "label_clustering", "label_random", "label_majority", "label_centroid"]:
        if col in df.columns:
            cols.append(col)
    return cols


def evaluate(pred, gold, data=None):
    cluster_evaluator = ClusterEvaluator(pred, gold, data=data)
    cluster_external = cluster_evaluator.get_external_metrics()
    prec, rec, f1, support = precision_recall_fscore_support(gold, pred, average='weighted')
    acc = accuracy_score(gold, pred)
    kappa = cohen_kappa_score(gold, pred, weights='quadratic')

    if data is not None:
        cluster_evaluator.get_internal_metrics()
        cluster_internal = cluster_evaluator.get_internal_metrics()
    else:
        cluster_internal = {}

    return {
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'kappa': kappa,
        'accuracy': acc,
        **cluster_external,
        **cluster_internal,
    }


def evaluate_file(pred_path, embeddings_fp=None):
    pred_df = pd.read_csv(pred_path)
    gold_column = get_gold_column(pred_df)
    pred_columns = get_pred_columns(pred_df)
    embeddings = np.load(embeddings_fp) if embeddings_fp is not None else None

    return {
        pred_column: evaluate(pred_df[pred_column], pred_df[gold_column], embeddings) for pred_column in pred_columns
    }


if __name__ == '__main__':
    args = parse_args()

    pred_files = args.pred

    if args.embeddings is None:
        embeddings_fps = [None] * len(pred_files)
    elif len(args.embeddings) == 1:
        embeddings_fps = args.embeddings * len(pred_files)
    elif len(args.embeddings) == len(pred_files):
        embeddings_fps = args.embeddings
    else:
        raise ValueError("Number of embeddings files must match number of predictions files.")

    file_evaluations = []

    for pred_fp, embeddings_fp in zip(pred_files, embeddings_fps):
        file_evaluation = evaluate_file(pred_fp, embeddings_fp=embeddings_fp)
        for pred_name, eval_scores in file_evaluation.items():
            file_evaluation_df = pd.DataFrame([eval_scores])
            file_evaluation_df['prediction'] = pred_name
            file_evaluation_df['file'] = pred_fp.replace('.csv', '')
            file_evaluations.append(file_evaluation_df)

    res_df = pd.concat(file_evaluations)
    res_df.set_index(['file', 'prediction'], inplace=True)
    res_df.to_csv(args.output)
