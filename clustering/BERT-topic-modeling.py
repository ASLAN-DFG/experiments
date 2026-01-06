from tqdm import tqdm
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from shutil import copyfile

train_names, train_texts = [], []
for f in tqdm(list(os.listdir('E:/Temp/train'))[:]):
    train_names.append(f.replace('.txt', ''))
    train_texts.append(open('E:/Temp/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': train_names, 'text': train_texts})


model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(train_text_df['text'], show_progress_bar=True)


umap_embeddings = umap.UMAP(n_neighbors=15,
                            n_components=5,
                            metric='cosine').fit_transform(embeddings)


cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                          metric='euclidean',
                          cluster_selection_method='eom').fit(umap_embeddings)

import matplotlib.pyplot as plt

# Prepare data
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

'''
clustered_sentences = [[] for i in range(result['labels'].size)]
for sentence_id, cluster_id in enumerate(result['labels']):
    clustered_sentences[cluster_id].append(train_text_df['id'][sentence_id])


path = 'E:/PhD/FeedbackPrize/feedback-prize-2021/'
for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print("Size ", len(cluster))
    for id in cluster:
        copyfile(path+'train/'+id+'.txt',path+'clusters/'+str(i)+'/'+id+'.txt')
result = pd.concat([train_text_df, result['labels']], axis=1)
result.to_csv(path+"clustering_result.csv")
'''

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.savefig('clustering.png')

