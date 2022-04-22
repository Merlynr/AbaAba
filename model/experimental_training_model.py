# %% 环境定义
import time
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import random
from tensorflow import keras

start = time.time()

GAMMA = 0.99
BATCH_SIZE = 128  # 衰减因子
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 0.05  # 动作噪声方差

TRAIN_EPISODE = 6000


# %% DDPG

class DDPG():
    def __init__(self, gamma):
        '''分别定义衰减因子，环境交互次数记录变量，经验池以及大小，并定义
                相应的网络以及各自的学习率。'''

        self.gamma = gamma
        self.steps = 0
        self.replay_memory = deque(maxlen=1000000)
        self.actor = self.Actor()
        self.critic = self.Critic()

        # 創建Actor，Critic的副本
        self.actor_ = self.Actor()
        self.critic_ = self.Critic()

        # 设置两个网络的学习率
        self.optimizer_actor = optimizers.Adam(learning_rate=0.001)
        self.optimizer_critic = optimizers.Adam(learning_rate=0.001)

    '''创建Actor网络,此网络用于生成连续的取值,也就是连续区间的动作值
        最后一层使用"tanh"作为激活函数,把输出映射到(0-1)的连续区间上去
        Actor网络的输入是环境的观测值,也就是所表征环境状态的一组数据'''

    def Actor(self):
        model = tf.keras.Sequential([
            layers.Dense(100, activation='relu', input_shape=(1, 4)),
            layers.Dense(1, 'tanh')
        ])
        return model

    '''创建Critic网络,此网络用来"评价"动作网络执行动作的好坏
        区别于上面的Actor网络,Critic网络的输入为环境的数据以及对应的动作数据
        简单来说就是咱总得对应起来吧,打分是根据你的动作与当前环境的状态来进行的
        如果前面有个障碍你不跳起来,而是蹲下,那你自然做了一个坏的动作,就这么个意思'''

    def Critic(self):
        model = tf.keras.Sequential([
            layers.Dense(100, activation='relu', input_shape=(1, 5)),
            layers.Dense(1, None)
        ])
        return model

    '''add函数实现将一条交互序列添加到经验池中的功能，并将reward函数重新赋值（给reward负值便于加快模型的收敛）'''

    def add(self, obs, action, reward, n_s, done):
        reward = reward if not done else -10.0
        self.replay_memory.append((obs, action, reward, n_s, done))

    '''
        data函数实现从经验池中取出一个batch的数据的功能，这个batch将送入网络进行训练
        '''

    def data(self):
        batch = random.sample(self.replay_memory, 128)
        Obs, Action, Reward, N_s, Done = [], [], [], [], []
        for (obs, action, reward, n_s, done) in batch:
            Obs.append(obs)
            Action.append(action)
            Reward.append(reward)
            N_s.append(n_s)
            Done.append(done)
        return np.array(Obs).astype("float32"), np.array(Action).astype("float32"), np.array(Reward).astype(
            "float32"), np.array(N_s).astype("float32"), np.array(Done).astype("float32")

    '''用於測試'''

    def predict(self, obs):
        pass

    '''更新actor网络：使用tensorflow的梯度处理方法
       （1）将环境的数值送入actor网络，拿到输出的动作
       （2）将环境的数值与所对应的动作拼接，并送入critic网络，拿到Q
       （3）所拿到的Q相当于评论员给的分数，这个Q被拿来计算损失
       （4）由于梯度下降会最小化loss，但是我们要最大化Q，因此在计算loss的时候，将Q*（-1.0）就可以达到最大化Q的目的
       （5）然后绑定actor网络的变量，并完成一次梯度下降处理，更新actor网络'''

    def learn_actor(self, obs):
        with tf.GradientTape() as tape:
            action = self.actor(obs)
            action = tf.reshape(action, shape=(128, 1))
            obs = tf.reshape(obs, shape=(128, 4))
            concat = tf.concat([obs, action], axis=1)
            Q = self.critic(concat)
            loss = tf.reduce_mean(-1.0 * Q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grads, self.actor.trainable_variables))

    '''更新cirtic网络：更新critic网络的过程较actor网络复杂一些。
        这里用到了DDPG（大大屁股）里面的D。一开始创建的网络副本，就在这里被使用到。副本名副其实，与原网络有着完全一样的结构与参数，起一个对照学习的作用。就像老师引导你做事情，会教给你一些个东西，你不能完全的自学习，有指导可以少走弯路。
        具体的一些细节我有在其他的文章中提到，如感兴趣，可以去看一下我的其他的文章，比如说DQN，Q_learning等文章。
        （1）critic网络输入为一个连续的交互序列。包含了前一个环境与动作，下一个环境与动作，以及执行前一个动作是否导致了环境终止的变量terminal。
        （2）先根据actor（副本）计算出下一个动作的Q，然后用来计算Target_Q。
        （3）Target_Q的计算要用到terminal。如果当前的动作导致环境结束，则没有下一个动作，则 (1.0-terminal) * self.gamma * next_Q = 0，说明没有下一个动作的转移。否则就做和计算。
        （4）拿到Target_Q之后，就可以用来计算损失。还是按照之前的方法计算出前一个（或者说是当前的）状态的Q值，与Target_Q做均方差计算损失，并最小化这个损失，使得上一个（现在的）状态接近“老师”所教给你的。
        （5）最后梯度下降，更新critic网络。'''

    def learn_critic(self, obs, action, reward, next_obs, terminal):
        next_action = self.actor_(tf.reshape(next_obs, shape=(128, 4)))
        concat = tf.concat([tf.reshape(next_obs, shape=(128, 4)), next_action], axis=1)
        next_Q = self.critic_(concat)

        terminal = tf.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.gamma * next_Q

        with tf.GradientTape() as tape:
            concat = tf.concat([tf.reshape(obs, shape=(128, 4)), tf.reshape(action, shape=(128, 1))], axis=1)
            Q = self.critic(concat)
            loss_ = tf.reduce_mean(tf.reduce_sum(tf.square(Q - target_Q)))

        grads_ = tape.gradient(loss_, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(grads_, self.critic.trainable_variables))

    '''学习的主函数：
        （1）每进行一百次交互，复制一次副本
        （2）当经验池中存够足量的交互序列，便可执行函数更新网络参数。
        '''

    def learn(self):

        if self.steps % 100 == 0:
            self.actor_.set_weights(self.actor.get_weights())
            self.critic_.set_weights(self.critic.get_weights())

        if len(self.replay_memory) >= 1000:
            obs, action, reward, next_obs, terminal = self.data()
            actor_loss = self.learn_actor(obs)
            critic_loss = self.learn_critic(obs, action, reward, next_obs, terminal)

#%%

from cartpole_continuous import ContinuousCartPoleEnv

env = ContinuousCartPoleEnv()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(obs_dim, act_dim)

agent = DDPG(gamma=GAMMA)

Scores = []

# 训练正式开始
for times in range(6000):
    s = env.reset()
    Score = 0
    step = 0
    while True:
        step += 1
        agent.steps += 1
        s = np.expand_dims(s, axis=0)
        # 拿到动作，添加噪声（有随机探索的价值）
        a = agent.actor(s.astype('float32'))
        a = np.clip(np.random.normal(a, NOISE), -1.0, 1.0)[0][0]
        next_s, reward, done, _ = env.step(a)
        agent.add(s, a, REWARD_SCALE * reward, next_s, done)
        # 每隔五个step训练一次，提高训练效率
        if agent.steps % 5 == 0:
            agent.learn()
        Score += reward
        s = next_s
        # env.render()

        if done or step >= 200:
            Scores.append(Score)
            print('episode:', times, 'score:', Score, 'max:', np.max(Scores))
            break

    if np.sum(Scores[-5:]) / 5 >= 198.0:
        agent.actor.save("actor.h5")
        agent.critic.save("critic.h5")
        break

end = time.time()
running_time = end - start
print('time cost : %.5f sec' % running_time)
