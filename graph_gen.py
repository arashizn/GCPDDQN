import networkx as nx

class GraphGen(object): 
    def gen_graph(self, num_min, num_max):
        max_n = num_max
        min_n = num_min
        cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)

        if self.training_type == 'random':
            weights = {}
            for node in g.nodes():
                weights[node] = random.uniform(0,1)
        else:
            degree = nx.degree_centrality(g)
            maxDegree = max(dict(degree).values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node]
        nx.set_node_attributes(g, weights, 'weight')
        return g

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        for i in tqdm(range(1000)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)