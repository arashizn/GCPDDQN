import networkx as nx
import numpy as np

state_s = []
st = np.zeros((2,3))
g = nx.barabasi_albert_graph(n=10, m=4)
weight_node = {}
for node in g.nodes():
    weight_node[node] = np.random.uniform(0,1)
weight_edge = {}
for u,v in g.edges():
    weight_edge[u,v] = np.random.randint(10,20)  
color_node = {}
for node in g.nodes():
    color_node[node] = np.random.randint(0,2)  
nx.set_node_attributes(g, weight_node, 'weight')
nx.set_edge_attributes(g, weight_edge, 'weight')
nx.set_node_attributes(g, color_node, 'color')
state_s.append(g)
#print(np.array(nx.adjacency_matrix(g).todense()))
g.remove_node(1)
g1 = g.copy()

state_s.append([1,2])
state_s.append(g1)
#edges = zip(*g.edges())
#print(nx.get_node_attributes(g,'weight'))
print(np.array(nx.get_edge_attributes(g,'weight').values()))
#print(nx.get_node_attributes(g,'color'))
#print(np.array(nx.adjacency_matrix(g1).todense()))

#np = np.random.rand(4,3)
#np1 = {0: 0, 1:11,2:22,3:33}
#g2 = nx.barabasi_albert_graph(n=10, m=4)
#state_s.append(g2)
#print( state_s[0].nodes())
#s = np.array(state_s, dtype=object)
#print(s[0])
