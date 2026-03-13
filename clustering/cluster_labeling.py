import numpy as np
import pandas as pd


def majority_voting(cluster_labels, gold_labels):
    """Assigns the most frequent gold label in a cluster to all its members."""
    cluster_labels = np.array(cluster_labels)
    gold_labels = np.array(gold_labels)
    propagated_labels = np.empty_like(gold_labels)
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]

    for cluster_id in unique_clusters:
        mask = (cluster_labels == cluster_id)
        majority_label = pd.Series(gold_labels[mask]).mode()[0]
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


def controlled_random_assignment(cluster_labels, gold_labels):
    """
    Assigns labels to clusters such that the distribution of assigned
    labels matches the distribution of the gold labels.
    """
    cluster_labels = np.array(cluster_labels)
    gold_labels = np.array(gold_labels)
    propagated_labels = np.empty_like(gold_labels)

    # 1. Identify unique clusters (excluding noise -1)
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    num_clusters = len(unique_clusters)

    if num_clusters == 0:
        return propagated_labels

    # 2. Calculate the global distribution of gold labels
    labels, counts = np.unique(gold_labels, return_counts=True)
    probabilities = counts / len(gold_labels)

    # 3. Determine how many clusters each label gets
    # We multiply probability by cluster count and round
    cluster_counts = np.round(probabilities * num_clusters).astype(int)

    # Adjustment: Ensure the sum of cluster_counts equals num_clusters
    # (Fixes rounding errors)
    diff = num_clusters - np.sum(cluster_counts)
    if diff != 0:
        # Adjust the label with the highest frequency to absorb the rounding error
        cluster_counts[np.argmax(counts)] += diff

    # 4. Create the pool of labels to be assigned to clusters
    assignment_pool = []
    for label, count in zip(labels, cluster_counts):
        assignment_pool.extend([label] * count)

    # 5. Shuffle the pool and map to cluster IDs
    np.random.shuffle(assignment_pool)
    cluster_to_label = dict(zip(unique_clusters, assignment_pool))

    # 6. Propagate labels back to the original array
    for cluster_id, assigned_label in cluster_to_label.items():
        mask = (cluster_labels == cluster_id)
        propagated_labels[mask] = assigned_label

    return propagated_labels


def centroid_based(cluster_labels, embeddings, gold_labels):
    """Assigns the gold label of the sample closest to the cluster centroid."""
    cluster_labels = np.array(cluster_labels)
    gold_labels = np.array(gold_labels)
    propagated_labels = np.empty_like(gold_labels)
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]

    for cluster_id in unique_clusters:
        mask = (cluster_labels == cluster_id)
        cluster_embeddings = embeddings[mask]
        cluster_gold = gold_labels[mask]

        if cluster_embeddings.shape[0] == 0:
            continue

        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = np.argmin(distances)
        propagated_labels[mask] = cluster_gold[closest_idx]

    return propagated_labels


def test_all_labeling_methods():
    # Setup: 4 clusters, 12 samples total
    # Gold Distribution: 9 "True" (75%), 3 "False" (25%)
    cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    gold_labels = np.array([
        "T", "T", "F",  # C0: 66% T
        "T", "T", "T",  # C1: 100% T
        "T", "F", "F",  # C2: 33% T
        "T", "T", "T"  # C3: 100% T
    ])

    # Controlled Embeddings
    embeddings = np.array([
        [0, 0], [5, 5], [5, 5],  # C0: Mean [3.3, 3.3]. Closest are idx 1,2 (T)
        [15, 15], [10, 10], [15, 15],  # C1: Mean [13.3, 13.3]. Closest is idx 4 (T)
        [20, 20], [25, 25], [25, 25],  # C2: Mean [23.3, 23.3]. Closest are idx 1,2 (F)
        [35, 35], [30, 30], [35, 35]  # C3: Mean [33.3, 33.3]. Closest is idx 10 (T)
    ])

    print("--- Running Cluster Labeling Test ---")

    # 1. Test Majority Voting
    # C0->T, C1->T, C2->F, C3->T (Result: 3 T, 1 F)
    res_major = majority_voting(cluster_labels, gold_labels)
    assert res_major[0] == "T" and res_major[6] == "F", "Majority Voting Logic Failure"
    print("✅ Majority Voting Passed")

    # 2. Test Standard Random Assignment
    # This just ensures every sample in a cluster gets the same valid label
    res_random = random_assignment(cluster_labels, gold_labels)
    for cid in range(4):
        mask = (cluster_labels == cid)
        assert len(np.unique(res_random[mask])) == 1, f"Cluster {cid} not homogeneous"
        assert res_random[mask][0] in ["T", "F"], "Invalid label picked"
    print("✅ Standard Random Passed")

    # 3. Test Controlled Random Assignment
    # EXPECTATION: Exactly 3 clusters assigned "T", 1 cluster assigned "F"
    res_controlled = controlled_random_assignment(cluster_labels, gold_labels)

    # Extract the unique label assigned to each of the 4 clusters
    assigned_to_clusters = [
        res_controlled[0],  # Label for C0
        res_controlled[3],  # Label for C1
        res_controlled[6],  # Label for C2
        res_controlled[9]  # Label for C3
    ]

    t_count = assigned_to_clusters.count("T")
    f_count = assigned_to_clusters.count("F")

    print(f"Controlled Results: {assigned_to_clusters}")
    assert t_count == 3, f"Expected 3 'T' clusters, got {t_count}"
    assert f_count == 1, f"Expected 1 'F' cluster, got {f_count}"
    print("✅ Controlled Random Distribution Passed (75% vs 25%)")

    # 3. Centroid Based
    res_centroid = centroid_based(cluster_labels, embeddings, gold_labels)

    # In my revised C2 embeddings:
    # Points: [20,20], [25,25], [25,25] -> Mean: [23.33, 23.33]
    # Dist to [20,20]: 4.71
    # Dist to [25,25]: 2.35 (Indices 1 and 2 of this cluster, which are global idx 7 and 8)
    # gold_labels[7] and gold_labels[8] are both "F"
    assert res_centroid[6] == "F", f"Centroid failed: Expected F, got {res_centroid[6]}"
    print("✅ Centroid Based Passed")


# Run the test
test_all_labeling_methods()
