import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra


def perform_pca(data, num_components=50):
    # Center the data
    data_mean = torch.mean(data, dim=0)
    centered_data = data - data_mean

    # Compute the covariance matrix
    cov_matrix = torch.matmul(centered_data.T, centered_data) / (data.size(0) - 1)

    # Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]

    # Project the data onto the top principal components
    pca_data = torch.matmul(centered_data, top_eigenvectors)

    return pca_data, top_eigenvectors


def geodesic_distance_matrix(data, k_neighbors, norm=2):
    N = data.size(0)

    # Step 1: Calculate pairwise Euclidean distances
    distances = torch.cdist(data, data, p=norm)

    # Step 2: Build the k-NN graph (using a list of edges with uniform weights)
    _, neighbors = torch.topk(distances, k=k_neighbors + 1, largest=False)
    adjacency_matrix = torch.full((N, N), float("inf"))  # Initialize with "Inf"
    for i in range(N):
        adjacency_matrix[i, i] = 0  # Distance to itself is 0
        for j in neighbors[i][1:]:
            adjacency_matrix[i, j] = 1  # Use uniform weight of 1
            adjacency_matrix[j, i] = 1  # Undirected graph

    # Step 3: Compute shortest paths using Dijkstra's algorithm
    geodesic_distances = dijkstra(adjacency_matrix.numpy(), directed=False)

    return geodesic_distances


def calculate_betti_numbers(dgm_array, threshold, dim):
    betti_numbers = {}

    # Iterate over each dimension present in the data
    complexity = 0
    for k in range(dim + 1):
        # Filter features of dimension k
        features_k = dgm_array[dgm_array[:, 2] == k]

        # Count features that are alive at the threshold (death > threshold)
        alive_features = np.sum((features_k[:, 0] <= threshold) & (features_k[:, 1] > threshold))
        complexity += alive_features

        # Store the Betti number for dimension k
        betti_numbers[f"beta_{k}"] = alive_features

    betti_numbers["complexity"] = complexity

    return betti_numbers
