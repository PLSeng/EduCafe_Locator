import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def _closest_centroid(self, sample):
        distances = [self.euclidean_distance(sample, point) for point in self.centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
        
    def fit(self, X, plot_steps=False):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        # optimization
        for i in range(self.max_iters):
            clusters = [[] for _ in range(self.K)]
            for idx, sample in enumerate(self.X):
                # find the closest centroid to the sample
                closest_centroid = self._closest_centroid(sample)
                clusters[closest_centroid].append(idx)
                
            # check if the clusters have changed
            if self.clusters == clusters:
                break
            
            self.clusters = clusters
            
            # update centroids
            for idx, cluster in enumerate(self.clusters):
                cluster_mean = np.mean(self.X[cluster], axis=0)
                self.centroids[idx] = cluster_mean
                
            if plot_steps:
                self.plot(title=f"Iteration {i}")
                
    def plot(self, title = 'KMeans Clustering'):
        # generate 100 random colors as list
        colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, self.K)]
        
        for idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                plt.scatter(self.X[sample_idx][0], self.X[sample_idx][1], color=colors[idx], edgecolor='k')
        for centroid in self.centroids:
            plt.scatter(centroid[0], centroid[1], s=130, color='black', marker='x')
        plt.title(title)
        plt.show()

    def calculate_sse(self):
        sse = 0
        for idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                sse += (self.euclidean_distance(self.X[sample_idx], self.centroids[idx]))**2
        return sse

def elbow_plot(X, k_max=10):
    sse = []
    for k in range(1, k_max+1):
        kmeans = KMeans(K=k)
        kmeans.fit(X)
        sse.append(kmeans.calculate_sse())
    plt.plot(range(1, k_max+1), sse)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()
