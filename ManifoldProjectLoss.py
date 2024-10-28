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
    MANIFOLD_RATIO = None
    CLOSING_MANIFOLD_RATIO = None
    DIM = None
    def __init__(self, data,k_neighbor=5, manifold_ratio = 0.3, closing_manifold_ratio = 0.1, device=None, **kwargs):
        super().__init__()
        self.n_data, self.d = data.shape
        if device is None:
            device = torch.device("cpu")
        adj_sparse = get_adjancency_matrix(data, k_neighbor, **kwargs)
        ManifoldProjection.DIM = self.d
        G: nx.Graph = nx.from_scipy_sparse_array(adj_sparse, create_using=nx.Graph)
        self.G = G
        min_dist = adj_sparse.data[adj_sparse.data > EPS].min()
        self.min_dist = min_dist
        ManifoldProjection.MANIFOLD_RATIO = manifold_ratio
        ManifoldProjection.CLOSING_MANIFOLD_RATIO = closing_manifold_ratio
        self.data = data.to(device)
        self.n = G.number_of_nodes()
        self.cache = {}

    class ProjectionFunction(torch.autograd.Function):
        """
        Custom autograd function to handle the manifold projection
        and its gradient computation
        """

        @staticmethod
        def forward(ctx, x, y, proj_x, proj_y, neighbor_x, neighbor_y):
            """
            Forward pass computes ||y - π(x)||²
            Args:
                x: Input point
                y: Target point
                projection_fn: Function that projects point onto manifold
            """

            # Store values for backward pass
            ctx.save_for_backward(x, y, proj_x, proj_y, neighbor_x, neighbor_y)

            return (y-x).norm()**2

        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass computes gradient:
            ∂L/∂x = 2(z - π(x)) * grad_output
            """
            x, y, proj_x, proj_y, neighbor_x, neighbor_y = ctx.saved_tensors
            if (neighbor_x-proj_x).norm() < EPS:
                return 2*(x-y)*grad_output, 2*(y-x)*grad_output,None,None, None, None
            manifold_normalized_grad_x = -( (neighbor_x - proj_x) / (neighbor_x-proj_x).norm())
            manifold_normalized_grad_y = -((neighbor_y - proj_y) / (neighbor_y - proj_y).norm())
            closing_manifold_grad_x = (x-proj_x) / (x-proj_x).norm()
            closing_manifold_grad_y = (y-proj_y) / (y-proj_y).norm()
            grad_dir_x = ManifoldProjection.CLOSING_MANIFOLD_RATIO * closing_manifold_grad_x+ ManifoldProjection.MANIFOLD_RATIO * manifold_normalized_grad_x + ( 1 - ManifoldProjection.MANIFOLD_RATIO - ManifoldProjection.CLOSING_MANIFOLD_RATIO) * (x-y) / (x-y).norm()
            grad_dir_y =  ManifoldProjection.CLOSING_MANIFOLD_RATIO * closing_manifold_grad_y+ ManifoldProjection.MANIFOLD_RATIO *  manifold_normalized_grad_y + ( 1 - ManifoldProjection.MANIFOLD_RATIO - ManifoldProjection.CLOSING_MANIFOLD_RATIO) * (y-x) / (y-x).norm()
            grad_x = 2*(x-y).norm()* grad_dir_x / grad_dir_x.norm()
            grad_y = 2*(y-x).norm()*grad_dir_y / grad_dir_y.norm()
            return grad_x*grad_output, grad_y*grad_output, None, None, None, None

    def clear_cache(self):
        self.cache = {}
    def append_cache(self, encoded_id, shortest_path):
        assert len(shortest_path) >= 1, "No shortest path error"
        if len(shortest_path) == 1:
            id_closest_x, id_closest_y = shortest_path[0], shortest_path[0]
        else:
            id_closest_x, id_closest_y = shortest_path[1], shortest_path[-2]
        self.cache[encoded_id] = (id_closest_x, id_closest_y)



    def get_geodesic_neighbor(self, idx,idy):
        encoded_id = idx + self.n * idy
        if encoded_id not in self.cache:
            shortest_path = nx.dijkstra_path(self.G, int(idx), int(idy))
            self.append_cache(encoded_id, shortest_path)
        return self.cache[encoded_id]


    def dist_path(self,v):
        paths = {v: [v]}
        return paths,_dijkstra(self.G, v, _weight_function(self.G, "weight"), paths=paths)
    def precompute_all_cache(self):

        for idx in self.G:
            paths,di_test = self.dist_path(idx)
            for idy,shortest_path in paths.items():
                encoded_id = idx + self.n * idy
                self.append_cache(encoded_id,shortest_path)


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

        id_closest_x, id_closest_y = self.get_geodesic_neighbor(idx, idy)
        return self.ProjectionFunction.apply(x, y, self.data[idx], self.data[idy], self.data[id_closest_x], self.data[id_closest_y])



class BatchingCostModule(torch.nn.Module):
    def __init__(self, cost_module):
        super().__init__()
        self.cost_module = cost_module
    def forward(self, x, y):
        batch_size, N, _ = x.shape
        _, M, _ = y.shape
        assert batch_size == 1, "Do not work for multi batching"

        costs = torch.zeros(batch_size, N, M, device=x.device, dtype=x.dtype)

        # Calculate the cost for each (i, j) pair across N and M
        for i in range(N):
            for j in range(M):
                # Compute the cost between x[:, i, :] and y[:, j, :] for each batch
                costs[:, i, j] = self.cost_module(x[0, i, :], y[0, j, :]).view(1,-1)

        return costs

def line_data(x,y, n):
    data1 = torch.tensor(uniform_line(np.array(x + 0.5), np.array((y - 0.5)), n))
    data2 = torch.tensor(uniform_line(np.array([0.2, -1]), np.array([0.8, 0.5]), n // 2))
    data3 = torch.tensor(uniform_line(np.array([0.8, 0.5]), np.array([0, 1.5]), n // 2))
    data = torch.concatenate([data1, data2, data3], dim=0)
    return data



def try_main():
    import matplotlib.pyplot as plt
    from gen_geometry import uniform_circ
    n = 50
    lr = 0.01
    device = "cpu"
    print("device:", device)

    # Initialize points and ensure they require gradients
    x_start_np = np.array([3, -1.5])
    y_start_np = np.array([-1, 2])

    x = torch.tensor(x_start_np).to(device).requires_grad_(True)
    y = torch.tensor(y_start_np).to(device)

    # Generate manifold data
    print(type(x_start_np))
    data = torch.tensor(uniform_circ(200))

    # Create manifold projection instance
    manifold_proj = ManifoldProjection(data, device=device)

    # Setup optimizer
    optimizer = torch.optim.SGD([x], lr=lr)

    # Number of iterations for gradient descent
    num_iterations = 600
    history_x = []
    history_y = []

    # Gradient descent loop
    for i in range(num_iterations):
        optimizer.zero_grad()

        # Compute loss using manifold projection
        loss = manifold_proj(x, y)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        # Store current position for visualization
        history_x.append(x.detach().cpu().numpy().copy())
        history_y.append(x.detach().cpu().numpy().copy())
        # Optional: Print progress
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}, Loss: {loss.item():.6f}")

    # Visualization
    history_x = np.array(history_x)
    colors = plt.get_cmap("coolwarm")(np.linspace(0, 1, len(history_x)))

    plt.figure(figsize=(10, 8))
    # Plot manifold points
    plt.scatter(data[:, 0], data[:, 1], c="green", label="Manifold", alpha=0.5)

    # Plot optimization trajectory
    plt.scatter(history_x[:, 0], history_x[:, 1], c=colors, label="Trajectory")

    # Plot start and end points
    plt.scatter(y_start_np[0], y_start_np[1], color="purple", s=100, label="Target")
    plt.scatter(x_start_np[0], x_start_np[1], color="black", s=100, label="Start")

    plt.legend()
    plt.title("Manifold Projection Gradient Descent")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    try_main()


