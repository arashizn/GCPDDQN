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
        return graph_init

    def step(self, state, act_node):
        net = state.copy()
        self.state_seq.append(net)
        self.act_seq.append(act_node)
        net.remove_node(act_node)
        residual_net = net.copy()
        reward = self.getreward(state, residual_net)
        self.reward_seq.append(reward)
        if self.isterminal(residual_net):
            return reward, residual_net, True
        else:
            return reward, residual_net, False


    def isterminal(self, state):
        edges_remain = state.edges()
        if len(edges_remain)==0:
            return True
        else:
            return False

    def getreward(self, net, residual_net):
        net = net.copy()
        residual_net = residual_net.copy()
        net_score = self.COOPScore(net)
        residual_net_score = self.COOPScore(residual_net)
        reward = net_score - residual_net_score
        return reward

    def COOPScore(self, net):
        node_score = sum(list(nx.get_edge_attributes(net,'weight').values()))
        node_neigh_score = 0
        for u,v in net.edges():
            node_neigh_score = node_neigh_score + (net.nodes[u]['weight']+net.nodes[v]['weight'])*(2**net.edges[u,v]['weight'])
        coopscore = node_score + node_neigh_score
        return coopscore


