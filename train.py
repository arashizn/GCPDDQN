"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
"""

from dqn import DQNPrioritizedReplay
from uavs_env import UAVSEnv
from swarm_net_gen import GraphGen
from graph import GraphSet
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Comparemethod import Choosemethod


UAV_NUM = 50
MEMORY_SIZE = 20000
EMBEDDING_SIZE = 32
EPISODE = 1500

env = UAVSEnv()
graphgen = GraphGen()

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=UAV_NUM, n_features=1, n_embedding = EMBEDDING_SIZE, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False,
    )

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=UAV_NUM, n_features=1, n_embedding = EMBEDDING_SIZE, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())

    
def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    episode_reward = []

    graphgen.gen_new_graphs(UAV_NUM, graph_num=1000, is_test=False)
    TrainSet = graphgen.TrainSet

    for i_episode in range(EPISODE):
        if i_episode and i_episode % 500 == 0:
            graphgen.gen_new_graphs(UAV_NUM, graph_num=1000, is_test=False)

        action_list = []
        g_sample= TrainSet.Sample()
        observation = env.state0(g_sample)
        ep_reward = 0
        j = 0
        while True:
            action = RL.choose_action(observation, action_list)
            action_list.append(action)
            reward, observation_, isterminal = env.step(observation, action)
            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            # if RL.epsilon >= 0.9:
            #     print('single step reward:', reward)

            observation = observation_
            total_steps += 1

            ep_reward += reward

            if isterminal:
                print('Episode:', i_episode, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % RL.epsilon)
                steps.append(j)
                episode_reward.append(ep_reward)
                episodes.append(i_episode)
                break

            j = j + 1
            
    RL.plot_cost()
                   
    return np.vstack((episodes, steps, episode_reward))
his_prio = train(RL_prio)
#his_natural = train(RL_natural)

# compare based on first success
#plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()

Degree = Choosemethod('degree')
Weight = Choosemethod('weight')
graphgen.gen_new_graphs(UAV_NUM, graph_num=100, is_test=True)
TestSet = graphgen.TestSet
def test(Method):
    steps = []
    episode_reward = []

    for graph in TestSet.graphset.values():
        observation = env.state0(graph)
        action_list = []
        ep_reward = 0
        j = 0
        
        while True:

            action = Method.choose_action(observation, action_list)
            action_list.append(action)
            reward, observation_, isterminal = env.step(observation, action)
            observation = observation_
            if isterminal:
                print('Method:', Method, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward)
                steps.append(j)
                episode_reward.append(ep_reward)
                break
            ep_reward += reward
            j = j + 1
    return np.vstack((steps, episode_reward))

test_degree = test(Degree)
test_weight = test(Weight)
test_rl = test(RL_prio)
plt.plot(np.arange(100), test_degree[0, :],c = 'r')
plt.plot(np.arange(100), test_weight[0, :],c = 'g')
plt.plot(np.arange(100), test_rl[0, :],c = 'b')
plt.legend(['degree','weight','rl'], loc='best')
plt.ylabel('graph')
plt.xlabel('step')
plt.grid()
plt.show()

plt.plot(np.arange(100), test_degree[1, :]/test_degree[0, :],c = 'r')
plt.plot(np.arange(100), test_weight[1, :]/test_weight[0, :],c = 'g')
plt.plot(np.arange(100), test_rl[1, :]/test_rl[0, :],c = 'b')
plt.legend(['degree','weight','rl'], loc='best')
plt.ylabel('graph')
plt.xlabel('mean_reward')
plt.grid()
plt.show()



