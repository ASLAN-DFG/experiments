import matplotlib.pyplot as plt
import pandas as pd


def print_clusters(result, text_list):
    clustered_dict = {}
    for sentence_id, cluster_id in enumerate(result['labels']):
        if cluster_id not in clustered_dict:
            clustered_dict[cluster_id] = []
        clustered_dict[cluster_id].append(text_list[sentence_id])

    for cluster_id, sentences in clustered_dict.items():
        if cluster_id == -1:
            print(f"Cluster: Outliers")
        else:
            print(f"Cluster: {cluster_id}")
        print(f"Size: {len(sentences)}")
        print(f"First answer: {sentences}")
        print("-" * 20)


def visualize_clusters(result, title, fig_name):
    fig, ax = plt.subplots()
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=2, label='Outliers')
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=5, cmap='hsv_r')
    ax.set_title(title)
    plt.savefig(fig_name, dpi=300)
    plt.show()


def save_clustering_results(train_text_df, result, file_name):
    clustered_sentences = [[] for i in range(result['labels'].size)]
    for sentence_id, cluster_id in enumerate(result['labels']):
        clustered_sentences[cluster_id].append(train_text_df['Id'][sentence_id])

    # for i, cluster in enumerate(clustered_sentences):
        # print("Cluster ", i+1)
        # print("Size ", len(cluster))
        # for id in cluster:
            # copyfile(path+'train/'+id+'.txt',path+'clusters/'+str(i)+'/'+id+'.txt')
    result = pd.concat([train_text_df, result['labels']], axis=1)
    result.to_csv(file_name)