import numpy as np
import pandas as pd
from scipy import stats


def majority_voting(cluster_labels, gold_labels):
    """Assigns the most frequent gold label in a cluster to all its members."""
    propagated_labels = np.copy(cluster_labels).astype(float)
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
    propagated_labels = np.copy(cluster_labels).astype(float)
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]

    for cluster_id in unique_clusters:
        mask = (cluster_labels == cluster_id)
        # Pick one random sample's gold label from this cluster
        possible_labels = gold_labels[mask]
        random_label = np.random.choice(possible_labels)
        propagated_labels[mask] = random_label

    return propagated_labels


def centroid_based(self):
    """Assigns the gold label of the sample closest to the cluster centroid."""
    if self.embeddings is None:
        raise ValueError("Embeddings are required for centroid-based assignment.")

    propagated_labels = np.copy(self.cluster_labels).astype(float)

    for cluster_id in self.unique_clusters:
        mask = (self.cluster_labels == cluster_id)
        cluster_embeddings = self.embeddings[mask]
        cluster_gold = self.gold_labels[mask]

        # 1. Calculate centroid (mean vector)
        centroid = cluster_embeddings.mean(axis=0)

        # 2. Find index of sample closest to centroid (Euclidean distance)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = np.argmin(distances)

        # 3. Get the gold label of that central sample
        central_label = cluster_gold[closest_idx]
        propagated_labels[mask] = central_label

    return propagated_labels