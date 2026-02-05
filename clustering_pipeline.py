import pandas as pd
from clustering.clustering_evaluator import ClusterEvaluator
from clustering.clustering_engine import ClusterEngine
from clustering.clustering_util import print_clusters, visualize_clusters
from normalization.nomalization import keep_content_words_nltk
from args import parse_args


if __name__ == '__main__':
    args = parse_args()

    # Read data
    # TODO: Standard reader for general data form
    train_fp = 'data/'+ args.dataset+'/'+ args.train
    df_train = pd.read_csv(train_fp, index_col=0).drop_duplicates().dropna()
    text_list = df_train['EssayText'].tolist()

    # Normalization
    ## keep only content words ('N', 'V', 'J')
    text_list = keep_content_words_nltk(text_list)
    # TODO: Standard writer for each normalisation step (each step as a view for cas)

    # Normalization Evaluation


    # Clustering
    engine = ClusterEngine()
    #result, embeddings = engine.topic_modelling(text_list)
    result, embeddings = engine.k_means_clustering(text_list, 10)
    print_clusters(result, text_list)
    visualize_clusters(result, "results/Topic Clusters Visualization Content Words", 'results/topic_clustering_content.png')
    # TODO: Parameterization with argparse.ArgumentParser()

    # Clustering Evaluation
    clustering_evaluation = ClusterEvaluator(embeddings, result['labels'])
    clustering_evaluation.export_metrics()

    # Label Propagation
    # TODO: cluster_labeling.py

    # Scoring Evaluation
    # TODO: scores_data_frame in scoring/utils.py


