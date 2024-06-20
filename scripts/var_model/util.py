import torch
import networkx as nx
from basicts.utils.serialization import load_adj


class RandomWalkUti:

    @staticmethod
    def load_random_walk(path, mode="scalap"):
        _, adj = load_adj(path, mode)
        print(adj[26])
        rw = RandomWalkUti._calculate_random_walk_matrix(torch.Tensor(adj).to("cpu"))
        network = nx.from_numpy_array(rw)
        node_labels = {i: f'Node {i}' for i in range(len(network.nodes))}
        nx.set_node_attributes(network, node_labels, "label")
        return network, rw


    @staticmethod
    def _calculate_random_walk_matrix(adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(adj_mx.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(
            d_inv.shape).to(d_inv.device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx.numpy()