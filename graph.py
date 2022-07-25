import networkx as nx
import random

class Graph(object):

    def __init__(self, node_num, edge_num):
        self.node_num = node_num
        self.edge_num = edge_num
        self.edge_dict = {}
        self.adj_dict = {}
        self.node_weight = {}

    def graph(self, node_num, edge_num, edges_from, edges_to, weight_node, weight_edge):
        for i in range (1,node_num):
            x = edges_from[i]
            y = edges_to[i]
            w = weight_edge[i]
            self.adj_dict[x].append(y)
            self.adj_dict[y].append(x)
            self.edge_dict[i] = [x,y,w]
            self.node_weight[i] = weight_node[i]           

class GraphSet(object):

    def __init__(self):
        self.graphset = {}

    def InsertGraph(self, graphid, graph):
        self.graphset[graphid] = graph

    def Sample(self):
        sampleid = random.choice(list(self.graphset.keys())) 
        samplegraph = self.graphset[sampleid]
        return samplegraph

    def Clear(self):
        self.graphset.clear()

     

