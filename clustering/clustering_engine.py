import pandas as pd
import umap.umap_ as umap
import hdbscan
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional


class ClusterEngine:
    def __init__(
            self,
            model_name: str = 'distilbert-base-nli-mean-tokens',
            n_neighbors: int = 15,
            n_components: int = 5,
            min_cluster_size: int = 15
    ):
        self.model = SentenceTransformer(model_name)
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size

    def topic_modelling(self, text_list: List[str]) -> Tuple[pd.DataFrame, list]:
        """
        Method 1: HDBSCAN (Density-based)
        Uses the class-level n_neighbors, n_components, and min_cluster_size.
        """
        embeddings = self.model.encode(text_list, show_progress_bar=True)

        # Projection for clustering logic
        umap_embeddings = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric='cosine',
            random_state=42
        ).fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        ).fit(umap_embeddings)

        return self._prepare_output(embeddings, clusterer.labels_)

    def k_means_clustering(self, text_list: List[str], n_clusters: int) -> Tuple[pd.DataFrame, list]:
        """
        Method 2: K-Means (Centroid-based)
        Uses class-level n_neighbors and n_components for the pre-projection.
        """
        embeddings = self.model.encode(text_list, show_progress_bar=True)

        # Reduce dimensionality to improve K-Means performance
        clustering_input = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric='cosine',
            random_state=42
        ).fit_transform(embeddings)

        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(clustering_input)

        return self._prepare_output(embeddings, clusterer.labels_)

    def _prepare_output(self, embeddings, labels) -> Tuple[pd.DataFrame, list]:
        """Internal helper to generate the 2D visualization and return results."""
        viz_data = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=2,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        ).fit_transform(embeddings)

        result = pd.DataFrame(viz_data, columns=['x', 'y'])
        result['labels'] = labels
        return result, embeddings
