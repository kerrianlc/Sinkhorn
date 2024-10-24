import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function, _bellman_ford, _dijkstra
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy

import torch
import torch.nn as nn

EPS = 1e-6

from gen_geometry import uniform_circ, uniform_line


def johnson(G, weight="weight"):
    r"""Uses Johnson's Algorithm to compute shortest paths.

    Johnson's Algorithm finds a shortest path between each pair of
    nodes in a weighted graph even if negative weights are present.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    distance : dictionary
        Dictionary, keyed by source and target, of shortest paths.
    """
    dist = {v: 0 for v in G}
    pred = {v: [] for v in G}
    weight = _weight_function(G, weight)

    # Calculate distance of shortest paths
    dist_bellman = _bellman_ford(G, list(G), weight, pred=pred, dist=dist)

    # Update the weight function to take into account the Bellman--Ford
    # relaxation distances.
    def new_weight(u, v, d):
        return weight(u, v, d) + dist_bellman[u] - dist_bellman[v]

    def dist_path(v):
        paths = {v: [v]}
        return _dijkstra(G, v, new_weight, paths=paths)

    return {v: dist_path(v) for v in G}


def geo_dist(G, weight="weight", dtype=None):
    n = G.number_of_nodes()
    if dtype is None:
        dist_matrix = np.zeros((n, n))
    else:
        dist_matrix = np.zeros((n, n), dtype=dtype)

    dist = johnson(G, weight=weight)
    for vertex, dist_set in dist.items():
        for target, distance in dist_set.items():
            dist_matrix[vertex][target] = distance
    return dist_matrix


def get_adjancency_matrix(data: np.ndarray, k_neighbor, **kwargs):
    """

    :param data: (n_samples, n_features)
    :param n_neighbors: number of neighbors to connect to each element
    :param sigma: normalizing std for gaussian "reach"
    :param alpha:
    :param kwargs:
    :return:
    """
    neighborhood = NearestNeighbors(n_neighbors=k_neighbor, **kwargs).fit(data)
    # Step 2: Find the K nearest neighbors (including self)
    distances, indices = neighborhood.kneighbors(data)

    # Step 3: Construct the row, col, and data for the COO sparse matrix
    row_indices = np.repeat(np.arange(data.shape[0]), k_neighbor)  # Repeat each point index n_neighbors times
    col_indices = indices.flatten()
    # Step 4: Create a COO sparse adjacency matrix
    coo_adj_matrix = scipy.sparse.coo_matrix((distances.flatten(), (row_indices, col_indices)),
                                             shape=(data.shape[0], data.shape[0]))

    return coo_adj_matrix


def get_dense_adj(data):
    return cdist(data, data, "euclidean")


class ManifoldProjection(nn.Module):
    """
    Custom PyTorch module that implements projection onto a manifold with custom gradient computation.
    Computes forward: ||y - π(x)||² where y is the best neighbor.
    Backprop: 2(y - π(x))
    """

    def __init__(self, data,k_neighbor=5, device=None, **kwargs):
        super().__init__()
        self.n_data = data.shape[0]
        if device is None:
            device = torch.device("cpu")
        adj_sparse = get_adjancency_matrix(data, k_neighbor, **kwargs)

        G: nx.Graph = nx.from_scipy_sparse_array(adj_sparse, create_using=nx.Graph)
        self.G = G
        min_dist = adj_sparse.data[adj_sparse.data > EPS].min()
        self.min_dist = min_dist
        self.data = data.to(device)
        self.n = G.number_of_nodes()
        self.cache = {}

    class ProjectionFunction(torch.autograd.Function):
        """
        Custom autograd function to handle the manifold projection
        and its gradient computation
        """

        @staticmethod
        def forward(ctx, x, z, proj_x):
            """
            Forward pass computes ||y - π(x)||²
            Args:
                x: Input point
                y: Target point
                projection_fn: Function that projects point onto manifold
            """

            # Store values for backward pass
            ctx.save_for_backward(z, proj_x)

            # Compute ||y - π(x)||²
            diff = z - proj_x
            return torch.sum(diff **2)

        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass computes gradient:
            ∂L/∂x = 2(z - π(x)) * grad_output
            """
            z, proj_x = ctx.saved_tensors
            if (z-proj_x).norm() < EPS:
                return None,None,None
            grad_x = -1*(z - proj_x) / (z-proj_x).norm() * grad_output
            return grad_x, None, None

    def clear_cache(self):
        self.cache = {}
    def get_geodesic_neighbor(self, idx,idy):
        encoded_id = idx + self.n * idy
        if encoded_id not in self.cache:
            shortest_path = nx.dijkstra_path(self.G, int(idx), int(idy))
            if len(shortest_path) ==1:
                return shortest_path[0]
            assert len(shortest_path) >=1, "No shortest path error"
            id_closest = shortest_path[1]
            self.cache[encoded_id] = id_closest
        return self.cache[encoded_id]

    def get_proj(self, x):
        diff_x = self.data - x
        distances = torch.norm(diff_x, dim=1)
        return torch.argmin(distances)

    def forward(self, x, y):
        """
        Args:
            x: Input point
            y: Target point
        Returns:
            Scalar value ||z - π(x)||²
        """

        idx = self.get_proj(x)
        idy = self.get_proj(y)

        id_closest = self.get_geodesic_neighbor(idx, idy)

        return self.ProjectionFunction.apply(x, self.data[id_closest], self.data[idx])


