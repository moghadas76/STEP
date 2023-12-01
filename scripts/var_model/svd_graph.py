import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scripts.var_model.var_model import load_adj
# Create a sample graph (replace this with your graph)
_,G, net, _ = load_adj()

# Get the adjacency matrix
A = net
A = A - np.diag(np.diag(A))

# Apply SVD
k = 207  # Number of clusters
svd = TruncatedSVD(n_components=k)
svd_result = svd.fit_transform(A)
print(svd_result.shape)
breakpoint()
# Apply KMeans clustering on the reduced features
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(svd_result)

# Visualize the graph with node colors representing clusters
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=clusters, with_labels=True, cmap='viridis')
plt.show()