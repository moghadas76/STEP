import pytest
import networkx as nx
from STEP.step.step_arch.tsformer.mask import SpatialMaskGenerator
from STEP.basicts.utils import load_pkl


def test_knn_masking():
    adj = load_pkl("/home/seyed/PycharmProjects/step/STEP/datasets/METR-LA/adj_mx.pkl")
    net: nx.Graph = nx.from_numpy_array(adj[2])
    qq = net.subgraph(nodes=next(nx.connected_components(net)))
    m = SpatialMaskGenerator(207, 0.75)
    t_0, t_1 = m.knn_masking(qq)
    t_0, t_1 = set(t_0), set(t_1)
    assert len(t_0.union(t_1)) == 207
    assert t_0.intersection(t_1) == set()


if __name__ == '__main__':
    test_knn_masking()


