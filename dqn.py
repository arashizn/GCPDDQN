"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""
import numpy as np
import tensorflow as tf
import networkx as nx
import random


np.random.seed(1)
inf = 2147483647/2


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size),dtype = object), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            
class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            n_embedding,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_embedding = n_embedding
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, 3),dtype = object)

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        def build_layers(s, adj, c_names, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('l_emb'):
                w_emb = tf.get_variable('w_emb', [self.n_features, self.n_embedding], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b_emb = tf.get_variable('b_emb', [1, self.n_embedding], initializer=b_initializer, collections=c_names,  trainable=trainable)
                output = tf.matmul(s,w_emb)
                embedding_s = tf.nn.relu(tf.matmul(adj, output, a_is_sparse = True) + b_emb)
                embedding_avg_s = tf.reduce_mean(embedding_s, 0, keepdims = True)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_embedding, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(embedding_avg_s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,  trainable=trainable)
                out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input_state_feature
        self.adj = tf.placeholder(tf.float32, [None, None], name='adj')  # input_adj_matrix
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, self.adj, c_names, n_l1, w_initializer, b_initializer, True)

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        self.adj_ = tf.placeholder(tf.float32, [None, None], name='adj_')  # input_adj_matrix
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, self.adj_, c_names, n_l1, w_initializer, b_initializer, False)

    def store_transition(self, s, a, r, s_):
        transition = []
        transition.append(s)
        transition.append([a, r])
        transition.append(s_)
        transitionn = np.array(transition, dtype=object)
        if self.prioritized:    # prioritized replay
            #transition = np.hstack((s, [a, r], s_))
            self.memory.store(transitionn)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            #transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transitionn
            self.memory_counter += 1

    def laplacian_matrix_sys_normalized(self, s):
        adj_matrix = np.array(nx.adjacency_matrix(s).todense())
        #compute L=D^-0.5 * (A+I) * D^-0.5
        adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])

        exist = (adj_matrix > 0) * 1
        factor = np.ones(adj_matrix.shape[1])
        degree = np.dot(exist, factor)
        
        d_hat = np.diag(np.power(degree, -0.5).flatten())
        norm_adj = d_hat.dot(adj_matrix).dot(d_hat)
        return norm_adj, degree


    def choose_action(self, observation, steps):
        graph = observation.copy()
        remain_node = graph.nodes() #obtain the avaiable node of the residual net

        adj, degree = self.laplacian_matrix_sys_normalized(graph)
        # state_feature_w = np.transpose(np.matrix(list(nx.get_node_attributes(graph,'weight').values())))# feature matrix of the residual net
        # state_feature_d = np.transpose((np.matrix(degree) - min(degree))/(max(degree) - min(degree)))
        # state_feature = np.hstack((state_feature_w, state_feature_d))
        state_feature = np.transpose((np.matrix(degree) - min(degree))/(max(degree) - min(degree)))#使用度作为特征

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: state_feature, self.adj: adj})
            for node in steps:
                actions_value[0][node] = -inf
            action = np.argmax(actions_value)
        else:
            action = random.choice(list(remain_node.keys()))
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        batch_s = batch_memory[:, 0]
        batch_s_ = batch_memory[:,2]
        cost = 0
        for i in range(self.batch_size):
            s = batch_s[i]
            s_ = batch_s_[i]
            adj, degree = self.laplacian_matrix_sys_normalized(s)
            adj_, degree_ = self.laplacian_matrix_sys_normalized(s_)
            # state_feature_w = np.transpose(np.matrix(list(nx.get_node_attributes(s,'weight').values())))
            # state_feature_d = np.transpose((np.matrix(degree) - min(degree))/(max(degree) - min(degree)))
            # state_feature = np.hstack((state_feature_w, state_feature_d))
            # state_feature_w_ = np.transpose(np.matrix(list(nx.get_node_attributes(s_,'weight').values())))
            # state_feature_d_ = np.transpose((np.matrix(degree_) - min(degree_))/(max(degree_) - min(degree_)))
            # state_feature_ = np.hstack((state_feature_w_, state_feature_d_))
            state_feature = np.transpose((np.matrix(degree) - min(degree))/(max(degree) - min(degree)))
            state_feature_ = np.transpose((np.matrix(degree_) - min(degree_))/(max(degree_) - min(degree_)))
            q_next, q_eval = self.sess.run(
                    [self.q_next, self.q_eval],
                    feed_dict={self.s_: state_feature_, self.adj_: adj_, 
                            self.s: state_feature, self.adj: adj})

            q_target = q_eval.copy()
            #batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[i,1][0]
            reward = batch_memory[i,1][1]

            q_target[0][eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

            if self.prioritized:
                _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                            feed_dict={self.s: state_feature,
                                                        self.adj: adj,
                                                        self.q_target: q_target,
                                                        self.ISWeights: ISWeights})
            else:
                _, self.cost = self.sess.run([self._train_op, self.loss],
                                            feed_dict={self.s: state_feature,
                                                        self.adj: adj,
                                                        self.q_target: q_target})
            cost  = cost + self.cost
        #print('loss is %7.2f' % cost)

        self.cost_his.append(cost)

        if self.prioritized:
            self.memory.batch_update(tree_idx, abs_errors)     # update priority

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
