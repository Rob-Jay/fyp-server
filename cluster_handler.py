from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import manifold
import numpy as np


class Clusterer:
    # Clustering algorithim created with PCA, KMaens a random state and a min centoid of .4e10
    def __init__(self, transformer: str = 'PCA', algorithm: str = 'KMeans', random_state=1234, centroid_min=.4e10):
        self.transformer = transformer
        self.algorithm = algorithm
        self.random_state = random_state
        self.centroid_min = centroid_min

#Selects model depending on the __init__. GaussianMixture or Kmeans will be selected
    def __get_model(self, k: int):
        if self.algorithm == 'gmm':
            return GaussianMixture(n_components=k, random_state=self.random_state)
        return KMeans(n_clusters=k, random_state=self.random_state)


#DifferentDimensionality reduction techniques
    def __get_transformer(self):
        if self.transformer == 'MDS':
            return manifold.Mds(n_components=2)
        elif self.transformer == 'TSNE':
            return manifold.TSNE(n_components=2)
        return PCA(2)

# Getting centroid depending on what algorithim is being used
    def __get_centroids(self, model):
        if self.algorithm == 'gmm':
            return model.means_
        return model.cluster_centers_


# Clustering algorithim
    def cluster(self, sentence_embeddings, k):
        # cluster_args  by k
        if k <= 0:
            return []
        cur_arg = -1
        args = {}
        used_idx = []
        transformer = self.__get_transformer()
        features = transformer.fit_transform(sentence_embeddings)
        model_cluster = self.__get_model(k)
        model = model_cluster.fit(features)
        centroids = self.__get_centroids(model)
        #Selecting each centroid of summarization
        for j, centroid in enumerate(centroids):
            for i, feature in enumerate(features):
                value = np.linalg.norm(feature - centroid)
                if value < self.centroid_min and i not in used_idx:
                    cur_arg = i
                    self.centroid_min = value
            used_idx.append(cur_arg)
            args[j] = cur_arg
            #Min distance value to search around.
            self.centroid_min = .4e10
            cur_arg = -1
        return args.values()
