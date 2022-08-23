import networkx as nx
import numpy as np
from batch import Batchgraph
import matplotlib.pyplot as plt

state_s = []
st = np.zeros((2,3))
g = nx.barabasi_albert_graph(n=10, m=4)
state_s.append(g)

weight_node = {}
for node in g.nodes():
    weight_node[node] = np.random.uniform(0,1)
weight_edge = {}
for u,v in g.edges():
    weight_edge[u,v] = np.random.randint(10,20)/20
color_node = {}
for node in g.nodes():
    color_node[node] = np.random.randint(0,2)  
nx.set_node_attributes(g, weight_node, 'weight')
nx.set_edge_attributes(g, weight_edge, 'weight')
nx.set_node_attributes(g, color_node, 'color')
#g1 = dgl.from_networkx(g, node_attrs = ['weight'])
g1 = g.copy()
g1.remove_node(1)
#g2 = dgl.from_networkx(g, node_attrs = ['weight'])
#state_s.append([1,2])
state_s.append(g1)

g2 = g.copy()
g2.remove_node(0)
g2.remove_node(7)
g2.remove_node(5)
state_s.append(g2)

# g3 = g.copy()
# g3.remove_node(7)
# state_s.append(g3)

# ba = Batchgraph(3, state_s)
# a,b = ba.batched_graph()
# print(a,b)
#batch_graph = dgl.batch(state_s)
#edges = zip(*g.edges())
# print(nx.get_edge_attributes(g,'weight'))
# print(nx.get_edge_attributes(g,'weight'))
#print(batch_graph)
plt.subplot(211)
node = [node for (node, val) in g2.degree()]
degree = [val for (node, val) in g2.degree()]
nx.draw_networkx(g2, with_labels=True)

# c= g2.degree()
# a,b = zip(*c)
# index = degree.index(max(degree))
# action = node[index]
# print(a)
# print(b)
# print(g2.degree())
# print(index)
# print(action)
# # print(weight.get)
# # print(max(weight,key=weight.get))
# print(nx.get_node_attributes(g,'weight').values())

i = 3
for sub in nx.connected_components(g2):
    plt.subplot(2, 2, i)
    subgraph = g2.subgraph(sub)
    nx.draw_networkx(subgraph, with_labels=True)
    i = i + 1
#print(g2.subgraph(sub))

plt.show()



