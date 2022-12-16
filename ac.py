"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.
The Cartpole example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import networkx as nx
from uavs_env import UAVSEnv
from swarm_net_gen import GraphGen

UAV_NUM = 100
EMBEDDING_SIZE = 32
EPISODE = 5000


OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()

GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 100
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0000001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
GLOBAL_STEP = []

N_S = 1
N_A = 100

graphgen = GraphGen()
graphgen.gen_new_graphs(UAV_NUM, graph_num=1, is_test=False)
TrainSet = graphgen.TrainSet

class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.adj_mat_fea = tf.placeholder(tf.float32, [None, N_S], 'S')
                # self.adj = self.laplacian_matrix_sys_normalized(self.s)
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.adj_mat_fea = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            emb_a = tf.layers.dense(self.adj_mat_fea, EMBEDDING_SIZE, tf.nn.relu, kernel_initializer=w_init, name='emba')
            emb_avg_a = tf.reduce_mean(emb_a, 0, keepdims = True)
            l_a = tf.layers.dense(emb_avg_a, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            emb_c = tf.layers.dense(self.adj_mat_fea, EMBEDDING_SIZE, tf.nn.relu, kernel_initializer=w_init, name='embc')
            embed_avg_c = tf.reduce_mean(emb_c, 0, keepdims = True)
            l_c = tf.layers.dense(embed_avg_c, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, steps):  # run by a local
        state_feature = np.transpose(np.matrix(list(nx.get_node_attributes(s,'weight').values())))# feature matrix of the residual net
        adj = self.laplacian_matrix_sys_normalized(s)
        adj_fea = adj*state_feature
        prob_weights = SESS.run(self.a_prob, feed_dict={self.adj_mat_fea: adj_fea})
        for node in steps:
                prob_weights[0][node] = 0
        prob_weights = prob_weights/sum(prob_weights.ravel())

        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def laplacian_matrix_sys_normalized(self, s):
        adj_matrix = np.array(nx.adjacency_matrix(s).todense())
        #compute L=D^-0.5 * (A+I) * D^-0.5
        adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])

        exist = (adj_matrix > 0) * 1
        factor = np.ones(adj_matrix.shape[1])
        degree = np.dot(exist, factor)
        
        d_hat = np.diag(np.power(degree, -0.5).flatten())
        norm_adj = d_hat.dot(adj_matrix).dot(d_hat)
        return norm_adj


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = UAVSEnv()
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP, GLOBAL_STEP
        total_step = 1
        buffer_s, buffer_adj_fea, buffer_a, buffer_r = [], [], [], []
        while not COORD.should_stop() and GLOBAL_EP < EPISODE:
            j = 0
            g_sample= TrainSet.Sample()
            observation = self.env.state0(g_sample)
            ep_r = 0
            action_list = []
            while True:
                # if self.name == 'W_0':
                #     self.env.render()
                action = self.AC.choose_action(observation, action_list)
                action_list.append(action)
                reward, observation_, isterminal = self.env.step(observation, action)
               
                # s_, r, done, info = self.env.step(a)
                state_feature = np.transpose(np.matrix(list(nx.get_node_attributes(observation,'weight').values())))# feature matrix of the residual net
                adj = self.AC.laplacian_matrix_sys_normalized(observation)
                adj_fea = adj*state_feature
                
                state_feature_ = np.transpose(np.matrix(list(nx.get_node_attributes(observation_,'weight').values())))# feature matrix of the residual net
                adj_ = self.AC.laplacian_matrix_sys_normalized(observation_)
                adj_fea_ = adj_*state_feature_

                ep_r += reward
                buffer_s.append(observation)
                buffer_adj_fea.append(adj_fea)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0 or isterminal:   # update global and assign to local net
                    if isterminal:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.adj_mat_fea: adj_fea_})
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_adj_fea, buffer_a, buffer_v_target = np.vstack(buffer_adj_fea), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.adj_mat_fea: buffer_adj_fea,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_adj_fea, buffer_a, buffer_r = [], [], [], []
                    self.AC.pull_global()

                observation = observation_
                j = j + 1
                total_step += 1
                if isterminal:
                    GLOBAL_STEP.append(j)
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| step: %i" % j,
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_STEP)
    plt.xlabel('epoch')
    plt.ylabel('Total moving step')
    plt.show()