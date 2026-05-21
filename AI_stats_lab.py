import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# ============================================================
# Question 1: Unlabeled Data and K-Means Clustering
# ============================================================

def load_iris_unlabeled(feature_indices=(0, 1)):
    iris = load_iris()

    X = iris.data[:, feature_indices]
    feature_names = [iris.feature_names[i] for i in feature_indices]

    return {
        "X": X,
        "feature_names": feature_names
    }


def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0, 1.0, std)

    X_scaled = (X - mean) / std

    return {
        "X_scaled": X_scaled,
        "mean": mean,
        "std": std
    }


def fit_kmeans(X, K, random_state=0, n_init=10):
    model = KMeans(
        n_clusters=K,
        random_state=random_state,
        n_init=n_init
    )

    model.fit(X)

    return {
        "centroids": model.cluster_centers_,
        "labels": model.labels_,
        "objective": model.inertia_,
        "n_iter": model.n_iter_
    }


def compute_kmeans_objective(X, centroids, labels):
    objective = 0.0

    for i in range(len(X)):
        centroid = centroids[labels[i]]
        distance_squared = np.sum((X[i] - centroid) ** 2)
        objective += distance_squared

    return objective


# ============================================================
# Question 2: Choosing K, Underfitting/Overfitting, and Outliers
# ============================================================

def evaluate_k_values(X, k_values, random_state=0, n_init=10):
    objectives = []
    relative_improvements = []

    previous_objective = None

    for k in k_values:
        result = fit_kmeans(
            X,
            K=k,
            random_state=random_state,
            n_init=n_init
        )

        current_objective = result["objective"]
        objectives.append(current_objective)

        if previous_objective is None:
            relative_improvements.append(0.0)
        else:
            improvement = (
                previous_objective - current_objective
            ) / previous_objective

            relative_improvements.append(improvement)

        previous_objective = current_objective

    return {
        "k_values": k_values,
        "objectives": objectives,
        "relative_improvements": relative_improvements
    }


def choose_elbow_k(k_values, objectives):
    if len(k_values) < 3:
        return k_values[0]

    x1, y1 = k_values[0], objectives[0]
    x2, y2 = k_values[-1], objectives[-1]

    max_distance = -1
    best_k = k_values[0]

    for i in range(1, len(k_values) - 1):
        x0 = k_values[i]
        y0 = objectives[i]

        numerator = abs(
            (y2 - y1) * x0
            - (x2 - x1) * y0
            + x2 * y1
            - y2 * x1
        )

        denominator = np.sqrt(
            (y2 - y1) ** 2
            + (x2 - x1) ** 2
        )

        distance = numerator / denominator

        if distance > max_distance:
            max_distance = distance
            best_k = k_values[i]

    return best_k


def cluster_size_summary(labels, K):
    summary = {}

    for i in range(K):
        summary[i] = int(np.sum(labels == i))

    return summary


def identify_outliers_by_distance(X, centroids, labels, top_n=5):
    distances = []

    for i in range(len(X)):
        centroid = centroids[labels[i]]
        distance_squared = np.sum((X[i] - centroid) ** 2)
        distances.append(distance_squared)

    distances = np.array(distances)

    sorted_indices = np.argsort(distances)[::-1]

    top_indices = sorted_indices[:top_n]
    top_distances = distances[top_indices]

    return {
        "indices": top_indices,
        "distances": top_distances
    }


def diagnose_clustering_fit(K, elbow_k):
    if K < elbow_k:
        return "underfitting"

    if K == elbow_k:
        return "good_fit"

    return "overfitting"


# ============================================================
# Question 3: Visualization
# ============================================================

def plot_unlabeled_data(X, feature_names=None, title="Unlabeled Data"):
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1])

    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    ax.set_title(title)

    return fig, ax


def plot_kmeans_clusters(
    X,
    labels,
    centroids,
    feature_names=None,
    title="K-Means Clusters"
):
    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels
    )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=200,
        marker='X'
    )

    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    ax.set_title(title)

    return fig, ax


def plot_elbow_curve(
    k_values,
    objectives,
    title="Elbow Method"
):
    fig, ax = plt.subplots()

    ax.plot(k_values, objectives, marker='o')

    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Objective value")
    ax.set_title(title)

    return fig, ax


if __name__ == "__main__":
    print("Implement all required functions.")
