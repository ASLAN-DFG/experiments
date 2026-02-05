from sklearn import metrics
import numpy as np
import pandas as pd


class ClusterEvaluator:
    def __init__(self, data, labels, gold_labels=None, ignore_noise=True):
        """
        Works for ANY clustering algorithm.

        data: Feature matrix (embeddings or coordinates).
        labels: Predicted cluster labels from ANY model.
        gold_labels: Ground truth labels (if available).
        ignore_noise: If True, excludes labels == -1 (common in density-based models).
        """
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.gold_labels = np.array(gold_labels) if gold_labels is not None else None

        # Filter noise if requested and if noise exists
        if ignore_noise and -1 in self.labels:
            mask = self.labels != -1
            self.data = self.data[mask]
            self.labels = self.labels[mask]
            if self.gold_labels is not None:
                self.gold_labels = self.gold_labels[mask]

    # --- Internal Metrics (No Ground Truth) ---
    def get_internal_metrics(self):
        # Silhouette requires at least 2 clusters to run
        n_clusters = len(set(self.labels))
        if n_clusters < 2:
            return {"Error": "Less than 2 clusters found. Internal metrics cannot be calculated."}

        return {
            "Silhouette_Score": metrics.silhouette_score(self.data, self.labels),
            "Calinski_Harabasz_Index": metrics.calinski_harabasz_score(self.data, self.labels),
            "Davies_Bouldin_Index": metrics.davies_bouldin_score(self.data, self.labels)
        }

    # --- External Metrics (With Ground Truth) ---
    def get_external_metrics(self):
        if self.gold_labels is None:
            return {"Error": "Gold labels not provided."}

        # Calculate Purity
        contingency_matrix = metrics.cluster.contingency_matrix(self.gold_labels, self.labels)
        purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        inv_purity = np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)

        # V-Measure components
        homo, comp, v = metrics.homogeneity_completeness_v_measure(self.gold_labels, self.labels)

        return {
            "Purity": purity,
            "Inverse_Purity": inv_purity,
            "Rand_Index": metrics.rand_score(self.gold_labels, self.labels),
            "Adjusted_Rand_Index": metrics.adjusted_rand_score(self.gold_labels, self.labels),
            "Adjusted_Mutual_Info": metrics.adjusted_mutual_info_score(self.gold_labels, self.labels),
            "Homogeneity": homo,
            "Completeness": comp,
            "V_Measure": v,
            "Fowlkes_Mallows": metrics.fowlkes_mallows_score(self.gold_labels, self.labels)
        }

    # --- Export to Dataframe ---
    def export_metrics(self):
        # Internal
        int_res = self.get_internal_metrics()
        df_int = pd.DataFrame([int_res])
        print("Internal Metrics:\n", df_int)

        # External
        df_ext = pd.DataFrame()
        if self.gold_labels is not None:
            ext_res = self.get_external_metrics()
            df_ext = pd.DataFrame([ext_res])
            print("\nExternal Metrics:\n", df_ext)

        return df_int, df_ext