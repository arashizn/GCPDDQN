import random
import networkx as nx
import numpy as np


class UAVSEnv(object):


    def __init__(self):
        self.state_seq = []
        self.act_seq = []
        self.reward_seq = []

    def state0(self,graph_init):
        self.state_seq.clear()
        self.act_seq.clear()
        self.reward_seq.clear()

    def step(self, state, act_node):
        net = state
        self.state_seq.append(net)
        self.act_seq.append(act_node)
        residual_net = net.remove_node(act_node)
        reward = getreward(net, residual_net)
        self.reward_seq.append(reward)

    def isterminal(self, state):
        edges_remain = state.edges()
        if len(edges_remain)==0:
            return True
        else:
            return False

    def getreward(self, net, residual_net):
        net_score = COOPScore(net)
        residual_net_score = COOPScore(residual_net)
        reward = net_score - residual_net_score
        return reward

    def COOPScore(self, net):
        node_score = sum(nx.get_edge_attributes(net,'weight').values())
        node_neigh_score = 0
        for u,v in net.edges():
            node_neigh_score = node_neigh_score + (graph.nodes[u]+graph.nodes[v])*(2**graph.edges[u,v]['weight'])
        coopscore = node_score + node_neigh_score
        return coopscore


