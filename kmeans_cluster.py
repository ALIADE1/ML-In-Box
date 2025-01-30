import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

np.random.seed(37)


class KMeans:
    def __init__(self, k=3, max_iteration=100):
        self.k = k
        self.max_iteration = max_iteration
        self.clusters = {}
        self.centroids = []

    def Initlize_centroids(self, data):
        rand_indic = np.random.choice(data.shape[0], self.k, replace=False)
        self.centroids = data[rand_indic]

    def Distance(self, point, centroid):
        return np.sqrt(np.sum((point - centroid) ** 2))

    def Assign_Clusters(self, data):
        # Reset clusters
        self.clusters = {i: [] for i in range(self.k)}

        for point in data:
            distances = [self.Distance(point, centroid) for centroid in self.centroids]
            cluster_idx = np.argmin(distances)
            self.clusters[cluster_idx].append(point)

    def Update_Centrloids(self):
        for i in range(self.k):
            self.centroids[i] = np.mean(self.clusters[i], axis=0)

    def Fit(self, data):
        self.Initlize_centroids(data)

        for _ in range(self.max_iteration):
            self.Assign_Clusters(data)
            prev_cent = np.copy(self.centroids)
            self.Update_Centrloids()

            if np.allclose(self.centroids, prev_cent, rtol=1e-3):
                break

    def Predict(self, data):
        predictions = []
        for point in data:
            distances = [self.Distance(point, centroid) for centroid in self.centroids]
            cluster_idx = np.argmin(distances)
            predictions.append(cluster_idx)
        return predictions


if __name__ == "__main__":
    data, _ = make_blobs(n_samples=500, centers=4, random_state=42, cluster_std=1.0)

    Kmeans = KMeans(k=4)
    Kmeans.Fit(data)

    predic_labels = Kmeans.Predict(data)

    for i in range(Kmeans.k):
        print(f"Cluster {i} has {len(Kmeans.clusters[i])} points")
        plt.scatter(np.array(Kmeans.clusters[i])[:, 0],np.array(Kmeans.clusters[i])[:, 1],label=f"Cluster {i}",)
    plt.legend()
    plt.show()
