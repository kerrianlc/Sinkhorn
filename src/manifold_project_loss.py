import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function, _bellman_ford, _dijkstra
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

EPS = 1e-6

from gen_geometry import uniform_circ, uniform_line





def get_adjancency_matrix(data: np.ndarray, k_neighbor, **kwargs):
    """

    :param data: (n_samples, n_features)
    :param n_neighbors: number of neighbors to connect to each element
    :param sigma: normalizing std for gaussian "reach"
    :param alpha:
    :param kwargs:
    :return: The adjacency matrix in scipy coo format.
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



def compute_vector_field(x, y, proj_x, neighbor_x, alpha,beta,gamma):
    manifold_normalized_grad = -((neighbor_x - proj_x) / (neighbor_x - proj_x).norm())
    closing_manifold_grad = (x - proj_x) / (x - proj_x).norm()
    vector_field = beta * closing_manifold_grad + gamma* manifold_normalized_grad + alpha * (x - y) / (x - y).norm()
    return vector_field


class ManifoldProjection(nn.Module):
    """
    Custom PyTorch module that implements projection onto a manifold with custom gradient computation.
    Computes forward: ||y - x||²
    Backprop: Vector Field Geodesic Aware normalized to have magnitude \|y-x \|.
    """
    MANIFOLD_RATIO = None # gamma parameter in the report
    CLOSING_MANIFOLD_RATIO = None # beta parameter in the report
    DIM = None # dimension of the data.
    def __init__(self, data,k_neighbor=5, manifold_ratio = 0.3, closing_manifold_ratio = 0.1, device=None, **kwargs):
        super().__init__()
        data = torch.Tensor(data)
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

    def draw_graph(self, node_color='blue', edge_color='black', node_size=50, figsize=(10, 10), title=None):
        """
        Draws the graph using the coordinates in data as node positions.

        Parameters:
            node_color (str or list): Color of the nodes (can pass a single color or a list matching the number of nodes).
            edge_color (str): Color of the edges.
            node_size (int): Size of the nodes in the visualization.
            figsize (tuple): Size of the figure (width, height).
            title (str): Title of the plot.
        """
        plt.figure(figsize=figsize)

        # Use `data` as the positions for the nodes
        pos = {i: self.data[i].cpu().numpy() for i in range(self.n)}
        assert pos[0].shape != 2, f" {pos[0].shape} is not (2,), data do not contain R^2 pointsn"

        # Take down self-loops
        edges_without_self_loops = [(u, v) for u, v in self.G.edges() if u != v]

        # Draw the nodes and edges
        nx.draw_networkx_nodes(self.G, pos, node_color=node_color, node_size=node_size, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, edgelist=edges_without_self_loops, edge_color=edge_color, alpha=0.5)

        # add title
        if title:
            plt.title(title, fontsize=16)

        # Display the plot
        plt.axis('off')
        plt.show()

    class ProjectionFunction(torch.autograd.Function):
        """
        Custom autograd function to have euclidian distance as forward propagation and
        handle the manifold aware vector field in backprop.
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
            Backward pass use vector field direction normalized to have magnitude \|y-x\|.
            """
            x, y, proj_x, proj_y, neighbor_x, neighbor_y = ctx.saved_tensors

            beta,gamma = ManifoldProjection.CLOSING_MANIFOLD_RATIO, ManifoldProjection.MANIFOLD_RATIO
            alpha = 1 - beta - gamma
            if (neighbor_x-proj_x).norm() < EPS:
                grad_x = 2*(x-y)*grad_output

            else:
                grad_dir_x = compute_vector_field(x,y,proj_x,neighbor_x, alpha,beta,gamma)
                grad_x = 2 * (x - y).norm() * grad_dir_x / grad_dir_x.norm()
            if (neighbor_y-proj_y).norm() < EPS:
                grad_dir_y = compute_vector_field(y,x,proj_y,neighbor_y, alpha,beta,gamma)
                grad_y = 2*(y-x).norm()*grad_dir_y / grad_dir_y.norm()

            else:
                grad_y = 2 * (y - x) * grad_output

            return grad_x*grad_output, grad_y*grad_output, None, None, None, None

    def clear_cache(self):
        """
        Clear the shortest path cache
        """
        self.cache = {}
    def append_cache(self, encoded_id, shortest_path):
        """
        Add to the cache the computed neighbors
        """
        assert len(shortest_path) >= 1, "No shortest path error" # The data graph is not connex
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
            Scalar value || y - x||² with custom backward geodesic aware vector field
        """

        idx = self.get_proj(x)
        idy = self.get_proj(y)

        id_closest_x, id_closest_y = self.get_geodesic_neighbor(idx, idy)
        return self.ProjectionFunction.apply(x, y, self.data[idx], self.data[idy], self.data[id_closest_x], self.data[id_closest_y])



class BatchingCostModule(torch.nn.Module):
    """
    Module that batch a given module. We need it to use our cost function with the Geomloss Sinkhorn function.
    """
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

        loss = manifold_proj(x, y)
        loss.backward()
        optimizer.step()

        history_x.append(x.detach().cpu().numpy().copy())
        history_y.append(x.detach().cpu().numpy().copy())

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


