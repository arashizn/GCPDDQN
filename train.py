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


UAV_NUM = 30
MEMORY_SIZE = 5000
EMBEDDING_SIZE = 32
EPISODE = 2000

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

    graphgen.gen_new_graphs(UAV_NUM)
    TrainSet = graphgen.TrainSet
    TestSet = graphgen.TestSet

    for i_episode in range(EPISODE):
        if i_episode and i_episode % 500 == 0:
                 graphgen.gen_new_graphs(UAV_NUM)

        action_list = []
        g_sample= TrainSet.Sample()
        observation = env.state0(g_sample)

        while True:
                action = RL.choose_action(observation, action_list)
                action_list.append(action)
                reward, observation_, isterminal = env.step(observation, action)
                RL.store_transition(observation, action, reward, observation_)

                if total_steps > MEMORY_SIZE:
                    RL.learn()

                observation = observation_
                total_steps += 1

                if isterminal:
                    print('episode ', i_episode, ' finished')
                    steps.append(total_steps)
                    episodes.append(i_episode)
                    break
                   
    return np.vstack((episodes, steps))

his_prio = train(RL_prio)
his_natural = train(RL_natural)

# compare based on first success
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()

