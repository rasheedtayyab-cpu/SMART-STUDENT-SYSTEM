from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_clustering(df):
    X = df.drop(columns=["FinalScore", "PassFail"])
    kmeans = KMeans(n_clusters=3, n_init=10)
    clusters = kmeans.fit_predict(X)

    reduced = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(reduced[:,0], reduced[:,1], c=clusters, cmap='viridis')
    plt.title("Student Clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig("outputs/clusters.png")
    plt.show()
    return clusters
