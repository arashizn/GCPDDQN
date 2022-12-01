import networkx as nx
import numpy as np
# g = nx.barabasi_albert_graph(n=10, m=3)
# a=np.array(list(g.nodes().keys()))
# print(a)
# b= np.arange(20)
# print(b)
# print(np.setdiff1d(b, a))
g = nx.read_gpickle('test50b1.gpickle')