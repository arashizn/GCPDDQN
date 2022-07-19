import networkx as nx
import numpy as np

state_s = []
g = nx.barabasi_albert_graph(n=10, m=4)
weight_node = {}
for node in g.nodes():
    weight_node[node] = np.random.uniform(0,1)
weight_edge = {}
for u,v in g.edges():
    weight_edge[u,v] = np.random.uniform(0,1)   
nx.set_node_attributes(g, weight_node, 'weight')
nx.set_edge_attributes(g, weight_edge, 'weight')

#edges = zip(*g.edges())
print(nx.get_edge_attributes(g,'weight'))
print(nx.get_node_attributes(g,'weight'))

g2 = nx.barabasi_albert_graph(n=10, m=4)
state_s.append(g2)
print( state_s[0])
