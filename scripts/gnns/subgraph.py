from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(dataset.num_node_features, 16)
        self.conv2 = GATv2Conv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, att = self.conv2(x, edge_index, return_attention_weights=True)
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(x, dim=1), att

# mask = torch.zeros(len(data.y)) + 1
# inds = torch.multinomial(mask, len(mask) * 0.2, replacement=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
# data = dataset[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# model.train()
# for epoch in range(1000):
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out, _ = model(data)
#         loss = F.nll_loss(out, data.y)
#         if epoch % 200 == 0:
#             print(loss)
#         loss.backward()
#         optimizer.step()
# torch.save(model.state_dict(), "/home/seyed/PycharmProjects/step/STEP/scripts/gnns/model.pt")
model.load_state_dict(torch.load("/home/seyed/PycharmProjects/step/STEP/scripts/gnns/model.pt"))
model.eval()
for data in loader:
    data = data.to(device)
    out, h = model(data)
    import networkx as nx

    # Assuming edge_list is your list of edges
    list_h = h[0].cpu().detach().numpy().T.tolist() # replace with your actual edge list
    att_list = h[1].cpu().detach().numpy().ravel().tolist()
    G = nx.Graph()
    # G.add_edges_from(list_h)
    for ind, edge in enumerate(list_h):
        if edge[0] != edge[1]:
            G.add_edge(edge[0], edge[1], attention="%.2f" % att_list[ind])
    node_labels = {i: f'{i}' for i in range(len(G.nodes))}
    nx.set_node_attributes(G, node_labels, "label")
    nodes_subset = list(range(0, 50))
    H = G.subgraph(nodes_subset)
    pos = nx.spring_layout(H)
    edge_labels = nx.get_edge_attributes(H, "attention")
    nx.draw(H, pos, with_labels=True, labels=nx.get_node_attributes(H, 'label'))
    nx.draw_networkx_edge_labels(H, pos, edge_labels)
    plt.show()
    break
    # tsne = TSNE(n_components=2, learning_rate='auto',
    #         init='pca').fit_transform(data.x.cpu().detach())

    # # Plot TSNE
    # plt.figure(figsize=(10, 10))
    # plt.axis('off')
    # plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y.cpu())
    # plt.show()
    # break 


# model.eval()
# pred = model(data).argmax(dim=1)
# correct = (pred == data.y).sum()
# acc = int(correct) / int(data.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')