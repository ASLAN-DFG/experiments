import math
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def nr_of_rows(df) -> int:
    return df.shape[0]


def contingency_matrix(df, column_cluster, column_category):
    cm = pd.crosstab(list(df[column_category]), list(df[column_cluster])).to_numpy(dtype=int)
    return cm


def purity_score(cm, inverse=False):
    return np.sum(np.max(cm, axis=1 if inverse else 0)) / np.sum(cm)


def precision(c, l):
    return len(c & l) / len(c)


def recall(l, c):
    return precision(c, l)


def f_measure(gold_labels, cluster_labels):
    gold_labels = list(gold_labels)
    cluster_labels = list(cluster_labels)
    assert len(gold_labels) == len(cluster_labels)
    c = [set([i for i, cl2 in enumerate(cluster_labels) if cl2 == cl]) for cl in set(cluster_labels)]
    l = [set([i for i, gl2 in enumerate(gold_labels) if gl2 == gl]) for gl in set(gold_labels)]
    n = len(gold_labels)

    return sum([len(li) * max([f_(li, cj) for cj in c]) / n for li in l])


def f_(l, c):
    rec = recall(l, c)
    prec = precision(l, c)
    return (2 * rec * prec) / (rec + prec) if rec + prec > 0 else 0


def count_pairs(a, b):
    ss = 0
    sd = 0
    ds = 0
    dd = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        p1 = a[i] == a[j]
        p2 = b[i] == b[j]
        if p1 and p2:
            ss += 1
        elif p1 and not p2:
            sd += 1
        elif not p1 and p2:
            ds += 1
        elif not p1 and not p2:
            dd += 1

    return ss, sd, ds, dd


def rand_statistic_r(a, b):
    ss, sd, ds, dd = count_pairs(a, b)

    return (ss + dd) / (ss + sd + ds + dd)


def jaccard_coefficient_j(a, b):
    ss, sd, ds, dd = count_pairs(a, b)

    return ss / (ss + sd + ds)


def folkes_mallows_fm(a, b):
    ss, sd, ds, dd = count_pairs(a, b)

    return math.sqrt((ss / (ss + sd)) * (ss / (ss + ds)))


def prob_cat_i_in_cluster_j(cm, i, j):
    return cm[i, j] / np.sum(cm[:, j])


def log2_(val):
    return math.log2(val) if val > 0 else 0


def cluster_entropy(cm, j):
    return sum([prob_cat_i_in_cluster_j(cm, i, j) * log2_(prob_cat_i_in_cluster_j(cm, i, j)) for i in range(cm.shape[0])])


def entropy(cm):
    return sum(np.sum(cm[:, j]) * cluster_entropy(cm, j) / np.sum(cm) for j in range(cm.shape[1]))


def pairwise_equal_mat(arr):
    n = len(arr)
    X = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if arr[i] == arr[j]:
                X[i, j] = 1
    return X


def pairwise_equal_mat_loose(arr: list[set]):
    n = len(arr)
    X = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if len(arr[i].intersection(arr[j])) != 0:
                X[i, j] = 1
    return X


def sim_correlation(embeddings, y, metric, name=""):
    gold_scores = pairwise_equal_mat(y)

    distances = -pairwise_distances(embeddings, embeddings, metric=metric)

    if metric == 'cosine':
        distances = distances + 1

    distances_ = list(zip(gold_scores.ravel(), distances.ravel()))
    pearson_r, _ = pearsonr(gold_scores.ravel(), distances.ravel())
    spearman_r, _ = spearmanr(gold_scores.ravel(), distances.ravel())
    gs_0_distances = [d for gs, d in distances_ if gs == 0]
    gs_1_distances = [d for gs, d in distances_ if gs == 1]
    # plt.violinplot([gs_0_distances, gs_1_distances], showmeans=True, showextrema=False)
    # plt.xticks([1, 2], ["0", "1"])
    # plt.title(name)
    # plt.savefig(f"fig/violin_corr_{name}.png")
    # plt.show()
    return pearson_r, spearman_r