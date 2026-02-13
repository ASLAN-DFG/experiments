import pandas as pd
from clustering.clustering_evaluator import ClusterEvaluator
from clustering.clustering_engine import ClusterEngine
from clustering.clustering_util import print_clusters, visualize_clusters
from normalization.nomalization import keep_content_words_nltk
from clustering.cluster_labeling import *
from data.data_utils.data_reader import AnswerReader
from scoring.scoring_utils import prepare_directories, scores_data_frame, save_data_frames
from args import parse_args


if __name__ == '__main__':
    args = parse_args()

    # Read data
    # TODO: Standard reader for general data form
    test_fp = 'data/' + args.dataset + '/' + args.test
    reader_test = AnswerReader(test_fp, args.dataset)
    df_test = reader_test.to_dataframe()

    text_list = df_test['EssayText'].tolist()
    score_list = np.array(df_test['Score1'].tolist(), dtype=int)

    # Normalization
    ## keep only content words ('N', 'V', 'J')
    text_list = keep_content_words_nltk(text_list)
    # TODO: Standard writer for each normalisation step (each step as a view for cas)

    # Normalization Evaluation
    # TODO


    # Clustering
    engine = ClusterEngine(args.clustering_method, args.n_clusters)
    result, embeddings = engine.run(text_list)
    #print_clusters(result, text_list)
    prepare_directories(args.experiment_name)
    visualize_clusters(result, args.dataset + ' ' + args.clustering_method, args.experiment_name + '/'+ args.dataset + '_' + args.clustering_method + '.png')

    # Clustering Evaluation
    clustering_evaluation = ClusterEvaluator(embeddings, result['labels'])
    clustering_evaluation.export_metrics()

    # Label Propagation
    # test_labeling_methods()
    label_random = random_assignment(result['labels'].tolist(), score_list)
    label_majority = majority_voting(result['labels'].tolist(), score_list)
    label_centroid = centroid_based(result['labels'].tolist(), embeddings, score_list)

    # Scoring Evaluation and save the results
    scores_random = scores_data_frame(score_list, label_random,'random')
    scores_majority = scores_data_frame(score_list, label_majority,'majority')
    scores_centroied = scores_data_frame(score_list, label_centroid,'centroid')

    output_dir = f"{args.experiment_name}/{args.dataset}_clustering"

    df_test['label_random'] = label_random
    df_test['label_majority'] = label_majority
    df_test['label_centroid'] = label_centroid
    combined_evaluation_df = pd.concat([scores_random, scores_majority, scores_centroied], ignore_index=True)

    save_data_frames(output_dir, [combined_evaluation_df, df_test], ['scores.csv','results.csv'])
    
