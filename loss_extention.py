
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function, _bellman_ford, _dijkstra
from scipy.spatial.distance import cdist
import numpy as np
from seaborn import husl_palette
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


def softmin(x, **kwargs):
    return torch.nn.functional.softmax(-1 * x, **kwargs)

class ExpandedGeodesicDist(nn.Module):
    def __init__(self, data, manifold_speed=2., k_neighbor=5, nb_softmin=10, device=None, **kwargs):

        self.n_data = data.shape[0]
        if device is None:
            device = torch.device("cpu")
        adj_sparse = get_adjancency_matrix(data, k_neighbor, **kwargs)


        G: nx.Graph = nx.from_scipy_sparse_array(adj_sparse, create_using=nx.Graph)

        min_dist = adj_sparse.data[adj_sparse.data > EPS].min()
        self.min_dist = min_dist
        self.data = data.to(device)
        self.manifold_dist = torch.tensor(geo_dist(G)).to(device)

        encoded_index_max_geodesic = self.manifold_dist.argmax()
        index_max_i = encoded_index_max_geodesic // self.n_data
        index_max_j = encoded_index_max_geodesic % self.n_data
        eucl_dist_max_geo = torch.linalg.vector_norm(data[index_max_i] - data[index_max_j])
        manifold_dist_dilatation = self.manifold_dist[index_max_i, index_max_j] / eucl_dist_max_geo
        self.manifold_speed = torch.tensor(manifold_speed).to(device)
        self.nb_softmin = nb_softmin
        self.manifold_dist_dilatation = manifold_dist_dilatation

    def get_knn(self, x):
        diff_x = self.data - x
        distances = torch.norm(diff_x, dim=1)
        closest_distances, indices_x = torch.topk(distances, self.nb_softmin, largest=False)
        return closest_distances, indices_x

    @staticmethod
    def cartesian_hadamard_product(x, y):
        return x[:, None] @ y[None, :]

    def forward(self, x, y):
        euc_dist_x, idx = self.get_knn(x)
        #euc_dist_y, idy = self.get_knn(y)
        #meshed = torch.meshgrid(idx, idy, indexing="ij")
        #manifold_dist_projected = self.manifold_dist[meshed]
        #prod_dist = self.cartesian_hadamard_product(euc_dist_x, euc_dist_y) / (50*self.min_dist**2)
        #attention_projected_coef = softmin(prod_dist.flatten(), dim=-1).reshape(self.nb_softmin, self.nb_softmin)
        #x_proj_attention = softmin(((1-attention_projected_coef) * manifold_dist_projected).flatten(), dim=-1).reshape(self.nb_softmin,self.nb_softmin).sum(dim=-1)

        #torch.dist(x,y) / self.manifold_speed
        return torch.dist(x,y) / self.manifold_speed + euc_dist_x.mean()


class GeoGrad():
    def __init__(self, data, manifold_speed=2., approaching_manifold_speed = 2.,k_neighbor=5, lr=0.01, device=None, **kwargs):

        self.n_data = data.shape[0]
        if device is None:
            device = torch.device("cpu")
        adj_sparse = get_adjancency_matrix(data, k_neighbor, **kwargs)


        G: nx.Graph = nx.from_scipy_sparse_array(adj_sparse, create_using=nx.Graph)
        self.G = G
        min_dist = adj_sparse.data[adj_sparse.data > EPS].min()
        self.min_dist = min_dist
        self.data = data.to(device)
        self.lr = lr
        self.manifold_speed = torch.tensor(manifold_speed).to(device)
        self.approaching_manifold_speed = torch.tensor(approaching_manifold_speed).to(device)
        self.n = G.number_of_nodes()
        self.cache = {}
    def clear_cache(self):
        self.cache = {}
    def get_geodesic_neighbor(self, idx,idy):
        encoded_id = idx + self.n * idy
        if encoded_id not in self.cache:
            shortest_path = nx.dijkstra_path(self.G, int(idx), int(idy))
            if len(shortest_path) <=1:
                return None
            id_closest = shortest_path[1]
            self.cache[encoded_id] = id_closest
        return self.cache[encoded_id]

    def get_proj(self, x):
        diff_x = self.data - x
        distances = torch.norm(diff_x, dim=1)
        return torch.argmin(distances)

    def __call__(self, x, y):
        idx = self.get_proj(x)
        idy = self.get_proj(y)


        id_closest = self.get_geodesic_neighbor(idx,idy)
        if id_closest is None:
            return -self.lr * (y-x)

        # torch.dist(x,y) / self.manifold_speed
        dir_euc = (y-x) / (y-x).norm()
        dir_manifold = (self.data[id_closest] - self.data[idx]) / (self.data[id_closest] - self.data[idx]).norm()
        dir_close_manifold = (self.data[id_closest] - x) / (self.data[id_closest] - x).norm()
        return -self.lr * (dir_euc + self.approaching_manifold_speed * dir_close_manifold +  self.manifold_speed * dir_manifold)


import matplotlib.pyplot as plt
def try_fun():
    n = 50
    lr=0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device : ", device)

    x_start = torch.tensor([0.4, -1.5])
    x = x_start.clone().detach().to(device)

    y = torch.tensor([0, 1.5]).to(device)

    data1 = torch.tensor(uniform_line(np.array(x_start + 0.5), np.array((y - 0.5).to("cpu")), n))
    data2 = torch.tensor(uniform_line(np.array([0.2,-1]), np.array([0.8,0.5]), n // 2))
    data3 = torch.tensor(uniform_line(np.array([0.8, 0.5]), np.array([0, 1.5]), n // 2))
    data = torch.concatenate([data1, data2, data3], dim=0)

    gradCompute = GeoGrad(data, lr = lr, manifold_speed=10, device=device)

    # Number of iterations for gradient descent
    num_iterations = 100
    history = []
    # Gradient descent loop
    for i in range(num_iterations):

        grad = gradCompute(x,y)
        x -= grad
        history.append(x.detach().to("cpu").numpy())
    history = np.array(history)
    print(history)
    colors = plt.get_cmap("coolwarm")(np.linspace(0, 1, len(history)))
    plt.scatter(data[:,0], data[:,1], c="green")
    plt.scatter(history[:, 0], history[:, 1], c=colors)
    plt.scatter(y[0].to("cpu").numpy(), y[1].to("cpu").numpy(), color="purple", s=100)
    plt.scatter(x_start[0], x_start[1], color="black", s=100)
    plt.show()


if __name__ == '__main__':
    try_fun()

