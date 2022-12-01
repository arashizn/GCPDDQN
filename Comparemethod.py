import networkx as nx

class Choosemethod(object):

    def __init__(self, method):
        self.method = method

    def choose_action(self, state, action_list = None):
        if self.method == 'degree':
            state_degree = state.degree()
            nodenum, degree = zip(*state_degree)
            action = nodenum[degree.index(max(degree))]
        elif self.method == 'weight':
            weight = nx.get_node_attributes(state,'weight')
            action = max(weight, key=weight.get)
        elif self.method == 'betweenness':
            betweenness = nx.betweenness_centrality(state)
            action = max(betweenness, key=betweenness.get)
        elif self.method == 'closeness':
            closeness = nx.closeness_centrality(state)
            action = max(closeness, key=closeness.get)
        elif self.method == 'eigenvector':
            eigenvector = nx.eigenvector_centrality(state)
            action = max(eigenvector, key=eigenvector.get)
        return action