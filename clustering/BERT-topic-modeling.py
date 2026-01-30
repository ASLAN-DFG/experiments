import argparse
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from shutil import copyfile
import matplotlib.pyplot as plt


def main(args):
    train_text_df = load_train_data(args.dataset)
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(train_text_df['EssayText'], show_progress_bar=True)

    embeddings_proj = apply_embedding_projection(embeddings, method=args.feature_projection)

    # cluster = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
    #                           metric='euclidean',
    #                           cluster_selection_method='eom').fit(embeddings_proj)
    cluster_size = args.cluster_size if args.cluster_size > 0 else len(set(train_text_df["Score1"]))
    cluster = KMeans(n_clusters=cluster_size, random_state=0).fit(embeddings_proj)

    # Prepare data
    umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    visualize_clusters(result)
    save_results(train_text_df, result)


def apply_embedding_projection(embeddings, method="umap"):
    if method == "umap":
        embeddings_proj = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine').fit_transform(embeddings)
    else:
        embeddings_proj = embeddings
    return embeddings_proj


def save_results(train_text_df: pd.DataFrame, result: pd.DataFrame):
    clustered_sentences = [[] for i in range(result['labels'].size)]
    for sentence_id, cluster_id in enumerate(result['labels']):
        clustered_sentences[cluster_id].append(train_text_df['Id'][sentence_id])

    path = 'out/'
    # for i, cluster in enumerate(clustered_sentences):
        # print("Cluster ", i+1)
        # print("Size ", len(cluster))
        # for id in cluster:
            # copyfile(path+'train/'+id+'.txt',path+'clusters/'+str(i)+'/'+id+'.txt')
    result = pd.concat([train_text_df, result['labels']], axis=1)
    result.to_csv(path+"clustering_result.csv")


def load_train_data(dataset: str):
    train_text_df = pd.read_csv(f'data/{dataset}')
    return train_text_df


def visualize_clusters(result: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.savefig('out/clustering.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="name of the dataset", default="ASAP/1_GOLD_test.csv")
    parser.add_argument("--cluster_size", "-cs", type=int, help="cluster size", default=-1)
    parser.add_argument("--feature_projection", "-fp", help="feature projection method", default="none", choices=["none", "umap"])
    args = parser.parse_args()
    main(args)
