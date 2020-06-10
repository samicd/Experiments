import math
import numpy as np
import scipy
import scipy.stats as stats
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('eigenStuf/Modi/Modi_expression_rma.txt', index_col='GeneSymbol', delimiter='\t')
df = df.drop(['LocusID', 'Description'], 1)

# We decide to include or remove a gene based on the standard deviation of the raw data.
raw_std = pd.DataFrame.std(df, axis=1)

# mean of the standard deviation, taken over all expression profiles:
mean_r_std = pd.DataFrame.mean(raw_std)

# variance of expression profile standard deviations as squared difference between raw_std and mean_r_std
var_of_std = sum((pow(x, 2) - pow(mean_r_std, 2)) for x in raw_std) / len(raw_std)

# We only want genes with raw_std > mean_r_std + k * sqrt(var_of_std)
rows_2_remove = []

# hyper parameter that defines the number of genes we use
k = 1

std_of_std = math.sqrt(var_of_std)
for gene, val in raw_std.iteritems():

    if val <= (mean_r_std + 0.9 * std_of_std):
        rows_2_remove.append(gene)

# Removing identified datum with low std of std value polished data = pol_p
polished_df = df.drop(rows_2_remove)

C = pd.DataFrame.cov(polished_df.transpose())

# z scores of the covariance matrix. Each profile has its mean value subtracted and then normalized to unit variance.
C = C.apply(scipy.stats.zscore)

# Calculate eigvals = eigenvalues of C
#           eigvec = eigenvectors of C
eigval, eigvec = np.linalg.eigh(C)

num_genes = polished_df.shape[0]

# generate matrix of zeros of the correct shape
M = np.zeros([num_genes, num_genes])

P = pd.DataFrame(np.linalg.pinv(C.values), C.columns, C.index)

P.index.name = None

links = P.stack().reset_index()
links.columns = ['var1', 'var2', 'value']

top_abs_links = links.reindex(links.value.abs().sort_values().index)

top_links_filtered = top_abs_links.loc[(top_abs_links['var1'] != top_abs_links['var2'])]
top100 = pd.DataFrame.tail(top_links_filtered, 100)

print(top100.tail(100))

G = nx.from_pandas_edgelist(top100, 'var1', 'var2','value')

# Currently this is an arbitrary value, I must check how they did it
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['value'] > 0]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['value'] < 0]

pos = nx.kamada_kawai_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=200)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,
                       width=1)
nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=1, alpha=0.5, edge_color='r', style='dashed')

# labels
nx.draw_networkx_labels(G, pos, font_size=7, font_family='sans-serif')

# nx.draw_kamada_kawai(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=10)
plt.show()
