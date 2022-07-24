"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
"""

import dqn
from uavs_env import UAVSEnv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

env = UAVSEnv()
uav_num = env.uav_num


MEMORY_SIZE = 10000
EMBEDDING_SIZE = 32
EPISODE = 2000

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=uav_num, n_features=1, n_embedding = EMBEDDING_SIZE, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False,
    )

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=uav_num, n_features=1, n_embeding = EMBEDDING_SIZE, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())

def fillmemory():
    
def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(EPISODE):
        observation = env.reset()
        while True:
            # env.render()

            action = RL.choose_action(observation, steps)

            observation_, reward, done, info = env.step(action)

            if done: reward = 10

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))

his_natural = train(RL_natural)
his_prio = train(RL_prio)

# compare based on first success
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()

