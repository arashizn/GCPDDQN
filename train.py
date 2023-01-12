"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
"""

from dqn_batch import DQNPrioritizedReplay
from uavs_env import UAVSEnv
from swarm_net_gen import GraphGen
from graph import GraphSet
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Comparemethod import Choosemethod
import os
import time
import networkx as nx


UAV_NUM = 150
MEMORY_SIZE = 10000
EMBEDDING_SIZE = 32
EPISODE = 600
G_TYPE = 'barabasi_albert'




env = UAVSEnv()
graphgen = GraphGen(G_TYPE)
graphgen.gen_new_graphs(UAV_NUM, graph_num=1, is_test=False)
TrainSet = graphgen.TrainSet

sess = tf.Session()

with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=UAV_NUM, n_features=2, n_embedding = EMBEDDING_SIZE, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.000032, sess=sess, prioritized=False,
    )

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=UAV_NUM, n_features=2, n_embedding = EMBEDDING_SIZE, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.000042, sess=sess, prioritized=True, output_graph=True,
    )
saver = tf.train.Saver(max_to_keep=None)   
sess.run(tf.global_variables_initializer())

 
def SaveModel(model_path):
    saver.save(sess, model_path)
    print('model has been saved success!')  

def loadBestModel():
    VCFile = './models/%s/ModelVC_%d.csv'%(G_TYPE, UAV_NUM)
    vc_list = []
    for line in open(VCFile):
        vc_list.append(float(line))
    best_model_iter = 20000 + 1000 * np.argmin(vc_list)
    best_model_path = './models/%s/nrange_%d_iter_%d.ckpt' % (G_TYPE, UAV_NUM, best_model_iter)

    saver.restore(sess, best_model_path)
    print('restore best model from file successfully')  


# TestSet = graphgen.TestSet
def test(Method):
    steps = []
    episode_reward = []
    start = time.time()

    for graph in TrainSet.graphset.values():
        observation = env.state0(graph)
        action_list = []
        ep_reward = 0
        j = 0
        
        while True:
            subgraphs = env.subgraph(observation)
            sub_coop = [env.COOPScore(g) for g in subgraphs]
            maxsub = subgraphs[sub_coop.index(max(sub_coop))]
            maxsubnode = np.array(list(maxsub.nodes().keys()))
            init = np.arange(UAV_NUM)
            infnode = np.setdiff1d(init,maxsubnode)

            action = Method.choose_action(observation, infnode)
            action_list.append(action)
            reward, observation_, isterminal = env.step(observation, action)
            observation = observation_
            if isterminal:
                print('Method:', Method.method, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward)
                steps.append(j)
                episode_reward.append(ep_reward)
                print(action_list)
                break
            ep_reward += reward
            j = j + 1
    end = time.time()
    time_consume = end - start
    print ('Method:', Method.method, 'testing 100 graphs time: %.2fs' %time_consume)
    return np.vstack((steps, episode_reward))

# graphgen.gen_new_graphs(UAV_NUM, graph_num=1, is_test=False)
# TrainSet = graphgen.TrainSet
# g = nx.read_gpickle('test50b1.gpickle')
# graphgen.insertgraph(g, is_test=False)
   
def train(RL):
    save_dir = './models/%s'%(G_TYPE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    VCFile = '%s/ModelVC_%d.csv'%(save_dir, UAV_NUM)
    f_out = open(VCFile, 'w')
    total_steps = 0
    steps = []
    episodes = []
    episode_reward = []


    for i_episode in range(EPISODE):
        # if i_episode and i_episode % 500 == 0:
        #     graphgen.gen_new_graphs(UAV_NUM, graph_num=1000, is_test=False)

        action_list = []
        g_sample= TrainSet.Sample()
        nx.write_gpickle(g_sample, "test.gpickle")
        observation = env.state0(g_sample)
        ep_reward = 0
        j = 0
        while True:
            subgraphs = env.subgraph(observation)
            sub_coop = [env.COOPScore(g) for g in subgraphs]
            maxsub = subgraphs[sub_coop.index(max(sub_coop))]
            maxsubnode = np.array(list(maxsub.nodes().keys()))
            init = np.arange(UAV_NUM)
            infnode = np.setdiff1d(init,maxsubnode)

            action = RL.choose_action(observation, infnode)
            action_list.append(action)
            reward, observation_, isterminal = env.step(observation, action)
            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            observation = observation_
            total_steps += 1

            ep_reward += reward

            if isterminal:
                print('Episode:', i_episode, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % RL.epsilon)
                steps.append(j)
                episode_reward.append(ep_reward)
                episodes.append(i_episode)
                if j <= 18:
                    print(action_list)
                    # model_path = '%s/nrange_%d_iter_%d.ckpt' % (save_dir, UAV_NUM, total_steps)
                    # SaveModel(model_path)
                
                    # test_tmp = test(RL_prio)
                    # f_out.write('%.16f\n' %(np.mean(test_tmp[0, :])))  #write vc into the file
                    # f_out.flush()
                break

            j = j + 1
    model_path = '%s/nrange_%d_iter_%d.ckpt' % (save_dir, UAV_NUM, total_steps)
    SaveModel(model_path)
    f_out.close()       
    RL.plot_cost()                  
    return np.vstack((episodes, steps, episode_reward))

his_prio = train(RL_prio)
np.savetxt("his_prio.txt", his_prio) 
#his_natural = train(RL_natural)
#np.savetxt("his_natural.txt", his_natural) 

# compare based on first success
#plt.plot(his_natural[0, :], his_natural[1, :], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total step')
plt.xlabel('episode')
plt.grid()
plt.show()

Degree = Choosemethod('degree')
Weight = Choosemethod('weight')
Betweenness = Choosemethod('betweenness')
Closeness = Choosemethod('closeness')
#Eigenvector = Choosemethod('eigenvector')

test_degree = test(Degree)
test_weight = test(Weight)
test_betweenness = test(Betweenness)
test_closeness = test(Closeness)
#test_eigenvector, time_e = test(Eigenvector)
# loadBestModel()
test_rl = test(RL_prio)
plt.plot(np.arange(1), test_degree[0, :],c = 'r')
plt.plot(np.arange(1), test_weight[0, :],c = 'g')
plt.plot(np.arange(1), test_betweenness[0, :],c = 'violet')
plt.plot(np.arange(1), test_closeness[0, :],c = 'gold')
#plt.plot(np.arange(100), test_eigenvector[0, :],c = 'k')
plt.plot(np.arange(1), test_rl[0, :],c = 'b')
plt.legend(['degree','weight','betweenness', 'closeness', 'rl'], loc='best')
plt.ylabel('total steps')
plt.xlabel('test graph')
plt.grid()
plt.show()

plt.plot(np.arange(100), test_degree[1, :]/test_degree[0, :],c = 'r')
plt.plot(np.arange(100), test_weight[1, :]/test_weight[0, :],c = 'g')
plt.plot(np.arange(100), test_betweenness[1, :]/test_betweenness[0, :],c = 'violet')
plt.plot(np.arange(100), test_closeness[1, :]/test_closeness[0, :],c = 'gold')
#plt.plot(np.arange(100), test_eigenvector[1, :]/test_eigenvector[0, :],c = 'k')
plt.plot(np.arange(100), test_rl[1, :]/test_rl[0, :],c = 'b')
plt.legend(['degree','weight','betweenness', 'closeness', 'rl'], loc='best')
plt.ylabel('mean reward')
plt.xlabel('test graph')
plt.grid()
plt.show()



