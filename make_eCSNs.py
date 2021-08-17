import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy  as np
import nx_pylabs
import glob

files = glob.glob('*.pkl')

for file in files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    
    for index in ['JT']:
        for c_threshold in [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            for edge_threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                try:
                    matrix = data[index][c_threshold]
                    matrix[matrix <= edge_threshold] = 0
                    g=nx.convert_matrix.from_numpy_matrix(matrix)
                    alpha = np.array([g[u][v]['weight'] for u,v in g.edges()])
                    alpha /= np.max(alpha)
                    
                    fig, ax = plt.subplots(1, 1, sharex=True)
                    x = nx_pylabs.draw_networkx(g, with_labels=True, edge_color=alpha, edge_cmap=plt.cm.Blues, pos=nx.drawing.layout.spring_layout(g, weight='weight', iterations=400, k=1/len(g)), ax=ax)
                    
                    name = file.split('_')[0]
                    name += '_' + index + '_c' + str(c_threshold) + '_e' + str(edge_threshold)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    plt.savefig(name + '.png')
                except ValueError:
                    break

    
