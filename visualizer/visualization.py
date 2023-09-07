import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from itertools import islice
from pathlib import Path
from basicts.utils import load_pkl

EPOCHS = 40
BATCH_NUM = 1627


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def generate_graph():
    adj = load_pkl("/home/seyed/PycharmProjects/step/STEP/datasets/METR-LA/adj_mx.pkl")
    net: nx.Graph = nx.from_numpy_array(adj[2])
    mapping: dict = adj[1]
    rev_mapping = {value: key for key, value in mapping.items()}
    distance = pd.read_csv("/STEP/datasets/METR-LA/distances_la_2012.csv")
    location = pd.read_csv("/STEP/datasets/METR-LA/graph_sensor_locations.csv")

    weights_map = {}
    # for _from in range(207):
    for (from_, to) in net.edges:
        try:
            weights_map[(from_, to)] = {
                "weight": distance.loc[(distance["from"] == int(rev_mapping[from_])) & (distance["to"] == int(rev_mapping[to]))].cost.values[0]
            }
        except (KeyError, IndexError) as err:
            logger.error(err)
            weights_map[(from_, to)] = {
                "weight": 0.0
            }

    labels = {
            i: {"label": str(tuple(
                location.loc[(location["sensor_id"] == int(rev_mapping[i]))][
                    ["latitude", "longitude"]].values.flatten().tolist()
            ))}
            for i in net.nodes
    }
    nx.set_edge_attributes(net, weights_map)
    nx.set_node_attributes(net, labels)
    # net = nx.relabel_nodes(
    #     net,
    #     labels
    # )
    path = "/STEP/checkpoints/"
    nx.write_gml(net, path+"metr-la.gml")
    with open(path+"labels.json", "w") as f:
        json.dump(labels, f)
    return net, labels

def visualize(net: nx.Graph, labels: dict):
    def chunk(arr_range, arr_size):
        arr_range = iter(arr_range)
        return iter(lambda: tuple(islice(arr_range, arr_size)), ())

    paths = sorted(Path("/STEP/loss").iterdir(), key=os.path.getmtime)[:65081]
    loss_ = []
    for epoch_files in chunk(paths, BATCH_NUM):
        n = np.zeros((1, 207))
        for file in list(epoch_files):
            try:
                np_arr = np.load(file.as_posix())["mae"][1, ...].reshape(1, 207)
                n = np.concatenate((n, np_arr))
            except:
                pass
        loss_.append(np.mean(n, axis=0))
    loss_ = np.array(loss_)
    x_ = np.arange(1, 41)
    nt = Network("1920px", "1080px", select_menu=True, filter_menu=True)
    for (src, to, val) in net.edges(data=True):
        nt.add_node(src, str(src), title=str(labels[src]["label"]))
        nt.add_node(to, str(to), title=str(labels[to]["label"]))
        nt.add_edge(src, to, value=val["weight"], title=str(val["weight"]))
    # nt.from_nx(net)

    plt.figure(figsize=(50, 50), dpi=450)
    for city in range(10):
        plt.plot(x_, loss_.T[city, :], label=f"City_{city}")
    plt.legend()
    plt.ylabel("loss")
    plt.ion()
    plt.show()
    plt.pause(10)
    nt.show(name="exp1.html", notebook=False)

def pars_args():
    parser = argparse.ArgumentParser(description="This is visualisation tool for traffic node monitoring")
    parser.add_argument("--load", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = pars_args()
    graph = None
    labels = None
    if args.load:
        graph = nx.read_gml(args.load)
        with open("/STEP/checkpoints/labels.json") as f:
            labels = json.load(f)
    else:
        graph, labels = generate_graph()
    visualize(graph, labels)
