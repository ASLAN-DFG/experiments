import numpy as np
import pandas as pd
from scipy import stats


def majority_voting(cluster_labels, gold_labels):
    """Assigns the most frequent gold label in a cluster to all its members."""
    cluster_labels = np.array(cluster_labels)
    gold_labels = np.array(gold_labels)
    propagated_labels = np.empty_like(gold_labels)
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]

    for cluster_id in unique_clusters:
        mask = (cluster_labels == cluster_id)
        # Find the mode (most common value) of gold labels in this cluster
        mode_result = stats.mode(gold_labels[mask], keepdims=True)
        majority_label = mode_result.mode[0]
        propagated_labels[mask] = majority_label

    return propagated_labels


def random_assignment(cluster_labels, gold_labels):
    """Assigns a random gold label existing within the cluster to all its members."""
    cluster_labels = np.array(cluster_labels)
    gold_labels = np.array(gold_labels)
    propagated_labels = np.empty_like(gold_labels)
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]

    for cluster_id in unique_clusters:
        mask = (cluster_labels == cluster_id)
        # Pick one random sample's gold label from this cluster
        possible_labels = gold_labels[mask]
        random_label = np.random.choice(possible_labels)
        propagated_labels[mask] = random_label

    return propagated_labels


def centroid_based(cluster_labels, embeddings, gold_labels):
    """Assigns the gold label of the sample closest to the cluster centroid."""
    cluster_labels = np.array(cluster_labels)
    gold_labels = np.array(gold_labels)
    propagated_labels = np.empty_like(gold_labels)
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        mask = (cluster_labels == cluster_id)
        cluster_embeddings = embeddings[mask]
        cluster_gold = gold_labels[mask]
        # Safety check: skip empty clusters (if any)
        if cluster_embeddings.shape[0] == 0:
            continue
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = np.argmin(distances)
        central_label = cluster_gold[closest_idx]
        propagated_labels[mask] = central_label

    return propagated_labels


def test_labeling_methods():
    # Setup test data
    # Cluster 0 has more "A" than "B"
    # Cluster 1 has more "B" than "A"
    cluster_labels = np.array([0, 0, 0, 1, 1, 1])
    gold_labels = np.array([
        "A", "A", "B", 
        "B", "B", "A"
    ])
    # Dummy embeddings for centroid (Center of C0 is index 1, C1 is index 4)
    embeddings = np.array([[0,0], [1,1], [2,2], [8,8], [9,9], [10,10]])

    # --- Test Majority Voting ---
    res_major = majority_voting(cluster_labels, gold_labels)
    assert res_major[0] == "A", "Majority Voting failed Cluster 0"
    assert res_major[3] == "B", "Majority Voting failed Cluster 1"
    print("Majority Voting Passed")

    # --- Test Random Assignment ---
    res_random = random_assignment(cluster_labels, gold_labels)
    # Check if the picked label is at least one of the valid options
    assert res_random[0] in ["A", "B"], "Random picked a label that didn't exist!"
    print(f"Random Assignment Passed (Picked: {res_random[0]} and {res_random[3]})")

    # --- Test Centroid Based ---
    # In C0, center is [1,1] which is label "A" (index 1)
    res_centroid = centroid_based(cluster_labels, embeddings, gold_labels)
    assert res_centroid[0] == "A", "Centroid Based failed"
    print("Centroid Based Passed")
