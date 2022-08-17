import networkx as nx
import numpy as np
from batch import Batchgraph

state_s = []
st = np.zeros((2,3))
g = nx.barabasi_albert_graph(n=5, m=4)
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
g2.remove_node(3)
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
#print(nx.get_node_attributes(g,'color'))
# adj_matrix = np.array(nx.adjacency_matrix(g1).todense())
# print(adj_matrix)
# exist = (adj_matrix > 0) * 1
# print(exist)
# factor = np.ones(adj_matrix.shape[1])
#res = np.dot(exist, factor)
# state_feature[node_feature.keys()] = node_feature.values()
# print(res[2])

#np = np.random.rand(4,3)
#np1 = {0: 0, 1:11,2:22,3:33}
#g2 = nx.barabasi_albert_graph(n=10, m=4)
#state_s.append(g2)
#print( state_s[0].nodes())
#s = np.array(state_s, dtype=object)
#print(s[0])
# def laplacian_martix_sys_normalized(s):
#     adj_matrix = np.array(nx.adjacency_matrix(s).todense())
#     #compute L=D^-0.5 * (A+I) * D^-0.5
#     adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])

#     exist = (adj_matrix > 0) * 1
#     factor = np.ones(adj_matrix.shape[1])
#     degree = np.dot(exist, factor)

#     #degree = np.array(adj_matrix.sum(1))
#     d_hat = np.diag(np.power(degree, -0.5).flatten())
#     norm_adj = d_hat.dot(adj_matrix).dot(d_hat)
#     return norm_adj

# a = laplacian_martix_sys_normalized(g)
# print(a)
def mean_matrix(sumnum):
        node_sum = sum(sumnum)
        mean_matrix = np.zeros((3, node_sum))
        for i in range(3):
            if i == 0 :
                mean_matrix[i,0:sumnum[i]] = 1/sumnum[i]
            else:
                mean_matrix[i,sum(sumnum[:i]):sum(sumnum[:i+1])] = 1/sumnum[i]
        return mean_matrix
sumnum = [2,3,5]
print(sumnum[:0])
print(mean_matrix(sumnum))

