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

        return action