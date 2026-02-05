from functools import reduce
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, accuracy_score


def compose(*fs):
    """
    Returns the composition of the given functions. Function outputs must be compatible with inputs of the next.
    Example:
        h(g(f(x))) == compose(f, g, h)(x)
    """

    def identity(x):
        return x

    def compose_pair(f, g):
        return lambda x: g(f(x))

    return reduce(compose_pair, fs, identity)


def prepare_directories(target_path):
    target_path = Path(target_path)
    output_directory = target_path / 'results'
    if not target_path.exists():
        target_path.mkdir()
    if not output_directory.exists():
        output_directory.mkdir()
    return target_path, output_directory


def determine_avg_type(labels):
    return 'macro' if len(labels) > 2 else 'binary'


def predictions_data_frame(y_true, y_predicted, dataset_name='NA', estimator_name='NA', feature_name='NA', fold=-1):
    data = [{
        'model': estimator_name,
        'features': feature_name,
        'dataset': dataset_name,
        'fold': fold,
        'gold': y_gold,
        'pred': y_hat} for y_gold, y_hat in zip(y_true, y_predicted)]
    index = ['model', 'features', 'dataset', 'fold']
    return pd.DataFrame(data).set_index(index)


def scores_data_frame(y_true, y_predicted, dataset_name='NA', estimator_name='NA', feature_name='NA', fold=-1, average_mode='weighted'):
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_predicted, average=average_mode)
    acc = accuracy_score(y_true, y_predicted)
    kappa = cohen_kappa_score(y_true, y_predicted, weights='quadratic')

    metric_names = ['precision', 'recall', 'f1_score', 'kappa', 'accuracy']
    metrics = [prec, rec, f1, kappa, acc]

    score = {metric: value for metric, value in zip(metric_names, metrics)}
    score['model'] = estimator_name
    score['features'] = feature_name
    score['dataset'] = dataset_name
    score['fold'] = fold

    data = [score]
    index = ['model', 'features', 'fold']
    return pd.DataFrame(data).set_index(index)


def save_data_frames(out_path, dfs, file_names):
    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir()

    for df, file_name in zip(dfs, file_names):
        csv_fp = out_path / file_name
        #df.to_csv(csv_fp)
        if csv_fp.exists():
            df.to_csv(csv_fp, mode='a', header=False)
        else:
            df.to_csv(csv_fp)
