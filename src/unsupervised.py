from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cluster_students(data):
    features = data.drop(columns=["FinalScore", "PassFail"])
    kmeans = KMeans(n_clusters=3, n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    reduced = PCA(n_components=2).fit_transform(features)
    plt.figure(figsize=(6,5))
    plt.scatter(reduced[:,0], reduced[:,1], c=cluster_labels, cmap='viridis')
    plt.title("Student Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig("outputs/student_clusters.png")
    plt.show()
    
    return cluster_labels
