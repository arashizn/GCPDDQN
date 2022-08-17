import networkx as nx
from graph import GraphSet
from graph import Graph
import sys 
import numpy as np
from tqdm import tqdm
import random
import math

DISTANCE_MIN = 5
DISTANCE_MAX = 20
DISTANCE_OPTIMAL = 10

class GraphGen(object): 
    def __init__(self):
        self.g_type = 'barabasi_albert'
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.training_type = 'random'   #'degree'
        self.TrainSet = GraphSet()
        self.TestSet = GraphSet()


    def gen_graph(self, num):
        #num = random.randint(num_min, num_max) 
        if self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=num, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=num, m=4)

        weight_node = {}
        if self.training_type == 'random':
            for node in g.nodes():
                weight_node[node] = random.uniform(0,1)
        else:
            degree = nx.degree_centrality(g)
            maxDegree = max(dict(degree).values())
            weight_node = {}
            for node in g.nodes():
                weight_node[node] = degree[node]

        weight_edge = {} 
        for u,v in g.edges():
            distance_edge = np.random.randint(DISTANCE_MIN,DISTANCE_MAX)
            if distance_edge <= DISTANCE_OPTIMAL:
                weight_edge[u, v] = 1
            else:
                weight_edge[u, v] = math.exp(-((distance_edge - DISTANCE_OPTIMAL)/5)) #weight of edge according to the distance between uav nodes
        nx.set_node_attributes(g, weight_node, 'weight')
        nx.set_edge_attributes(g, weight_edge, 'weight')
        return g

    def gen_new_graphs(self, num):
        print('\ngenerating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        for i in tqdm(range(1000)):
            g = self.gen_graph(num)
            self.insertgraph(g, is_test=False)
    
    def insertgraph(self,g,is_test):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, g)
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, g)

    def GenNetwork(self, g):    
        nodes = g.nodes()
        edges = g.edges()
        w_node = []
        w_edge = []

        for i in range(len(nodes)):
            w_node.append(g.nodes[i]['weight'])
        for i in range(len(edges)):
            w_edge.append(edges(i)['weight'])
        if len(edges) > 0:
            edgesfrom, edgesto = zip(*edges)
            edges_from = np.array(edgesfrom)
            edges_to = np.array(edgesto)
            weight_node = np.array(w_node)
            weight_edge = np.array(w_edge)
        else:
            edges_from = np.array([0])
            edges_to = np.array([0])
            weight_node = np.array([0])
            weight_edge = np.array([0])

        
        return Graph.graph_init(len(nodes), len(edges), edges_from, edges_to, weight_node, weight_edge)
    
    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()