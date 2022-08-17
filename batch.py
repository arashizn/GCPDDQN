import numpy as np
import networkx as nx

class Batchgraph(object):

    def __init__(self, batch_size, batch_graph):
        self.batch_graph = batch_graph
        self.batch_size = batch_size

    def Concat(self, a, b):
        lena = len(a)
        lenb = len(b)
        left = np.row_stack((a, np.zeros((lenb, lena))))  # 先将a和一个len(b)*len(a)的零矩阵垂直拼接，得到左半边
        right = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个len(a)*len(b)的零矩阵和b垂直拼接，得到右半边
        result = np.hstack((left, right))  # 将左右矩阵水平拼接
        return result
    
    def laplacian_martix(self, adj_matrix):
        adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])

        exist = (adj_matrix > 0) * 1
        factor = np.ones(adj_matrix.shape[1])
        degree = np.dot(exist, factor)

        #degree = np.array(adj_matrix.sum(1))
        d_hat = np.diag(np.power(degree, -0.5).flatten())
        norm_adj = d_hat.dot(adj_matrix).dot(d_hat)
        return norm_adj
        
    def batched_graph(self):
        batched_adj = np.empty((0,0))
        batched_feature = np.empty((0,1))
        subnum = []
        for i in range(self.batch_size):
            graph = self.batch_graph[i]
            node_num = len(graph.nodes())
            subnum.append(node_num)
            graph_feature = np.transpose(np.matrix((list(nx.get_node_attributes(graph,'weight').values()))))
            adj = np.array(nx.adjacency_matrix(graph).todense())

            batched_adj = self.Concat(batched_adj, adj)
            batched_feature = np.vstack([batched_feature, graph_feature])
        batched_la_adj = self.laplacian_martix(batched_adj)
        mean_matrix = self.mean_matrix(subnum)

        return batched_la_adj, batched_feature,mean_matrix
    
    def mean_matrix(self, subnum):
        node_sum = sum(subnum)
        mean_matrix = np.zeros((self.batch_size, node_sum))
        for i in range(self.batch_size):
            if i == 0 :
                mean_matrix[i,0:subnum[i]] = 1/subnum[i]
            else:
                mean_matrix[i,sum(subnum[:i]):sum(subnum[:i+1])] = 1/subnum[i]
        return mean_matrix



