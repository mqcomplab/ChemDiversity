import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy  as np


with open('ECFP4_CSN_data.pkl', 'rb') as f:
    data = pickle.load(f)

matrix = data['JT'][10]
matrix[matrix <= 0] = 0

g=nx.convert_matrix.from_numpy_matrix(matrix)
alpha = np.array([g[u][v]['weight'] for u,v in g.edges()])
alpha -= np.min(alpha)
alpha /= np.max(alpha)

fig, ax = plt.subplots(1, 1, sharex=True)
x = nx.drawing.nx_pylab.draw_networkx(g, with_labels=True, pos=nx.drawing.layout.spring_layout(g, weight='weight', iterations=400, k=1/len(g)), ax=ax)

edges = ax.collections[1]
#print(help(edges.set_alpha))
edges.set_alpha(alpha)

plt.show()
