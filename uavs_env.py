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
        self.state_init = graph_init
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
        # edges_remain = state.edges()
        # if len(edges_remain)==0:
        #     return True
        # else:
        #     return False
        si = self.COOPScore(self.state_init) 
        assert si != 0
        # s = self.COOPScore(state)
        sub = self.subgraph(state)
        sub_coop = [self.COOPScore(g) for g in sub]
        if (max(sub_coop)/si < 0.1):
            return True
        else:
            return False

    def getreward(self, net, residual_net):
        #si = self.COOPScore(self.state_init)
        # net_score = self.COOPScore(net)
        # residual_net_score = self.COOPScore(residual_net)
        # reward = net_score - residual_net_score

        sub = self.subgraph(residual_net)
        sub_coop = [self.COOPScore(g) for g in sub]
        # reward = np.std(sub_coop)
        reward = max(sub_coop)

        # if len(sub)==1:
        #     sub_coop.append(0)
        #     reward = np.std(sub_coop)

        sub2 = self.subgraph(net)
        sub_coop2 = [self.COOPScore(g) for g in sub2]
        # reward2 = np.std(sub_coop2)
        reward2 = max(sub_coop2)

        # if len(sub2)==1:
        #     sub_coop2.append(0)
        #     reward2 = np.std(sub_coop2)

        reward = reward2 - reward

        # if self.isterminal(residual_net):
        #     reward = reward + len(residual_net.nodes())*5
        # largest1 = max(nx.connected_components(residual_net),key=len)
        # largest_connected_subgraph1 = residual_net.subgraph(largest1)
        # largest2 = max(nx.connected_components(net),key=len)
        # largest_connected_subgraph2 = net.subgraph(largest2)
        # reward = self.COOPScore(largest_connected_subgraph2) - self.COOPScore(largest_connected_subgraph1)
        
        return reward

    def COOPScore(self, net):
        node_score = sum(list(nx.get_edge_attributes(net,'weight').values()))
        node_neigh_score = 0
        for u,v in net.edges():
            node_neigh_score = node_neigh_score + (net.nodes[u]['weight']+net.nodes[v]['weight'])*(2**net.edges[u,v]['weight'])
        coopscore = node_score + node_neigh_score
        return coopscore

    def subgraph(self, net):
        subgraphs = []
        for sub in nx.connected_components(net):
            subgraph = net.subgraph(sub)
            subgraphs.append(subgraph) 
        return subgraphs

        


