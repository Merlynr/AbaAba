import datetime

import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import random
import time
from collections import deque

from envir.GridMapEnv_2 import GridMapEnv, Points
from rich.progress import track

start_time = time.time()

from operator import itemgetter

episodes = 10000
episode_rewards = []
average_rewards = []
max_reward = 0
max_average_reward = 0
step_limit = 30
env = GridMapEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
template = 'episode: {}, rewards: {:.2f}, max reward: {}, mean_rewards: {}, epsilon: {}'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpus) > 0
tf.config.experimental.set_memory_growth(gpus[0], True)


class DQNAgent:
    def __init__(self):
        # other hyperparameters
        self.save_graph = True
        self.isTraining = True
        self.keepTraining = False
        self.play = False
        self.render = False
        self.save_model = True
        self.load_model = False
        self.random = False

        # epsilon greedy exploration
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.min_epsilon = 0.01
        self.linear_annealed = (self.initial_epsilon - self.min_epsilon) / 4000
        self.decay_rate = 0.995

        # check the hyperparameters
        if self.random == True:
            self.play = False
            self.isTraining = False
        if self.play == True:
            self.render = True
            self.save_model = False
            self.load_model = True
            self.isTraining = False
            self.keepTraining = False
        if self.keepTraining == True:
            self.epsilon = self.min_epsilon
            self.load_model = True
        # fixed q value - two networks
        self.learning_rate = 0.0001
        self.fixed_q_value_steps = 100
        self.target_network_counter = 0
        if self.load_model:
            self.model = keras.models.load_model('./model/Double_DQN/DDPG_model_sample.h5')
            self.target_model = keras.models.load_model('./model/Double_DQN/DDPG_target_sample_db.h5')
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()

        # experience replay
        self.experience_replay = deque(maxlen=3000)
        self.batch_size = 64
        self.gamma = 0.99
        self.replay_start_size = 320
        # 0.8-1.0极强相关，0.6-0.8强相关，0.4-0.6中等相关，0.2-0.4弱相关，0-0.2极弱或无相关相关，-1-0负相关，
        self.sample_proportion = [1, 3, 2.5, 1.5, 1, 1]

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_dim=state_size),
            keras.layers.Dense(action_size, activation='relu')]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss='mse',
                      metrics=['accuracy'])
        return model

    # def training(self, model):
    #     if len(self.experience_replay) >= self.replay_start_size:
    #         #if self.epsilon > self.min_epsilon:
    #         #    self.epsilon = self.epsilon * self.decay_rate
    #         batches = np.random.choice(len(self.experience_replay), self.batch_size)
    #         for i in batches:
    #             buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done = self.experience_replay[i]
    #             if buffer_done:
    #                 y_reward = buffer_reward
    #             else:
    #                 y_reward = buffer_reward + self.gamma*np.max(self.target_model.predict(buffer_next_state)[0])
    #             #only one output, which has the shape(1,2)
    #             y = model.predict(buffer_state)
    #             y[0][buffer_action] = y_reward
    #             model.fit(buffer_state, y, epochs=1, verbose=0)

    def training(self):
        if len(self.experience_replay) >= self.replay_start_size and len(self.experience_replay) % 32==0:
            # if self.epsilon > self.min_epsilon:
            #    self.epsilon = self.epsilon * self.decay_rate
            # 均匀随机采样
            # batches = random.sample(self.experience_replay, self.batch_size)
            # 根据优先级采样
            batches = self.sampleByResemblance()
            tem_batch_size = len(batches)
            buffer_state = [data[0] for data in batches]
            buffer_action = [data[1] for data in batches]
            buffer_reward = [data[2] for data in batches]
            buffer_next_state = [data[3] for data in batches]
            buffer_done = [data[4] for data in batches]

            buffer_state = np.reshape(buffer_state, (tem_batch_size, state_size))
            buffer_next_state = np.reshape(buffer_next_state, (tem_batch_size, state_size))
            y = self.model.predict(buffer_state)
            # DDQN double DQN: choose action first in current network,
            # no axis=1 will only have one value
            max_action_next = np.argmax(self.model.predict(buffer_next_state), axis=1)
            target_y = self.target_model.predict(buffer_next_state)
            for i in range(0, tem_batch_size):
                done = buffer_done[i]
                if not done:  # TODO 加了个not
                    y_reward = buffer_reward[i]
                else:
                    # then calculate the q-value in target network
                    target_network_q_value = target_y[i, max_action_next[i]]
                    y_reward = buffer_reward[i] + self.gamma * target_network_q_value
                # only one output, which has the shape(1,2)
                y[i][buffer_action[i]] = y_reward
            self.model.fit(buffer_state, y, epochs=64, verbose=0)

    def acting(self, state):
        if self.render:
            env.render()
        self.target_network_counter += 1
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
            print('weights updated')
        random_number = np.random.sample()
        if random_number > self.epsilon:
            action = np.argmax(self.model.predict(state)[0])
        else:
            action = np.random.randint(action_size)
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.linear_annealed
        return action

    # Since heavily importing into the global namespace may result in unexpected behavior,
    # the use of pylab is strongly discouraged
    def draw(self, episode, episode_rewards, average_rewards, location):
        plt.figure(figsize=(15, 6))
        plt.subplots_adjust(wspace=0.3)
        plt.subplot(1, 2, 1)
        # using polynomial to fit
        p1 = np.poly1d(np.polyfit(range(episode + 1), episode_rewards, 3))
        yvals = p1(range(episode + 1))
        plt.plot(range(episode + 1), yvals, 'b')
        plt.plot(range(episode + 1), episode_rewards, 'b')
        plt.title('score with episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.ylim(bottom=0)
        # average_rewards
        plt.subplot(1, 2, 2)
        plt.plot(range(episode + 1), average_rewards, 'r')
        plt.title('mean_score with episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.ylim(bottom=0)
        plt.savefig(location)
        plt.close()

    # 优先级
    def setResemblanceValue(self, example):
        point_a = np.array(example[5].split(':'),dtype=int)
        point_b = np.array(example[6].split(':'),dtype=int)

        key_words =  list(set(point_a).union(set(point_b)))

        point_a_vector = np.zeros(len(key_words))
        point_b_vector = np.zeros(len(key_words))

        # 计算词频率
        for i in range(len(key_words)):
            # 遍历key_word中每个词在句子中的出现次数
            for j in range(len(point_a)):
                if key_words[i] == point_a[j]:
                    point_a_vector[i] += 1
            for k in range(len(point_b)):
                if key_words[i] == point_b[k]:
                    point_b_vector[i] += 1

        pcc = float(np.dot(point_a_vector,point_b_vector)/(np.linalg.norm(point_a_vector)*np.linalg.norm(point_b_vector)))

        # l_a = len(point_a)
        # l_b = len(point_b)

        # 系数
        # a_b = 0.0

        # isLonger = True if l_a >= l_b else False
        # l = abs(l_a-l_b)
        # fixed_array = np.zeros((l,),dtype=int)
        # if isLonger:
        #     # point_a = point_a[0:l_b]
        #     # point_b.extend([0]*l)
        #     point_b=np.hstack((point_b,fixed_array))
        #     # point_b.append(np.zeros((l,),dtype=int))
        #     a_b = l_b / l_a
        # else:
        #     # point_b = point_b[0:l_a]
        #     # point_a.extend([0] * l)
        #     # np.hstack((point_a,(np.zeros((l,),dtype=int))))
        #     point_a=np.hstack((point_a, fixed_array))
        #     a_b = l_a / l_b

        # sum_ab = np.sum(np.sum(point_a,point_b))
        # sum_a = np.sum(np.sum(point_a))
        # sum_b = np.sum(np.sum(point_b))
        # sum_a2 = np.sum(np.sum(point_a*point_a))
        # sum_b2 = np.sum(np.sum(point_b*point_b))
        #
        # pcc = (l_a*sum_ab-sum_a*sum_b)/np.sqrt((l_a*sum_a2-sum_a*sum_a)*(l_a*sum_b2-sum_b*sum_b))

        # n = np.sum(a * b)
        # den = np.sqrt(np.sum(np.power(a, 2)) * np.sum(np.power(b, 2)))
        # 皮尔逊
        # pcc = np.corrcoef(point_a, point_b)
        # print(pcc)
        # 奖励等级 根据优先级进行优化
        reward = example[2]*pcc if (example[4] or (example[3][0][0]==example[3][0][2] and example[3][0][1]==example[3][0][3])) else env.punish[2]
        # if den == 0:
        #     return (example[0], example[1], reward, example[3], example[4], 0.0)
        return (example[0], example[1], reward, example[3], example[4], pcc if  example[4] else 0.0)

    # 排序
    def saveResemblanceValue(self):
        temporary_experience_replay = list(self.experience_replay)
        trends = sorted(temporary_experience_replay, key=itemgetter(5), reverse=True)
        return trends

    def sampleByResemblance(self):
        sum = self.sample_proportion[0] + self.sample_proportion[1] + self.sample_proportion[2] + \
              self.sample_proportion[3] + self.sample_proportion[4] + self.sample_proportion[5]
        trends = self.saveResemblanceValue()
        temporary_experience_replay = []
        # 直接通过大小进行判断划定范围
        trend_eps_1 = []
        trend_eps_2 = []
        trend_eps_3 = []
        trend_eps_4 = []
        trend_eps_5 = []
        trend_eps_6 = []
        # 对应范围样本不够，则不取。
        for value in trends:
            # print(value)
            if value[5] < 1 and value[5] >= 0.8:
                trend_eps_1.append(value)
            if value[5] < 0.8 and value[5] >= 0.6:
                trend_eps_2.append(value)
            if value[5] < 0.6 and value[5] >= 0.4:
                trend_eps_3.append(value)
            if value[5] < 0.4 and value[5] >= 0.2:
                trend_eps_4.append(value)
            if value[5] < 0.2 and value[5] >= 0.0:
                trend_eps_5.append(value)
            if value[5] < 0.0 and value[5] >= -1:
                trend_eps_6.extend(value)

        batch_size_1 = round(((self.sample_proportion[0] / sum) * self.batch_size))
        batch_size_2 = round(((self.sample_proportion[1] / sum) * self.batch_size))
        batch_size_3 = round(((self.sample_proportion[2] / sum) * self.batch_size))+1
        batch_size_4 = round(((self.sample_proportion[3] / sum) * self.batch_size))
        batch_size_5 = round(((self.sample_proportion[4] / sum) * self.batch_size))
        batch_size_6 = round(((self.sample_proportion[5] / sum) * self.batch_size))

        if len(trend_eps_1) > batch_size_1:
            temporary_experience_replay.extend(
                random.sample(trend_eps_1, batch_size_1))
        if len(trend_eps_1) > 0 and len(trend_eps_1) <= batch_size_1:
            temporary_experience_replay.extend(trend_eps_1)
        if len(trend_eps_2) > batch_size_2:
            temporary_experience_replay.extend(
                random.sample(trend_eps_2, batch_size_2))
        if len(trend_eps_2) > 0 and len(trend_eps_2) <= batch_size_2:
            temporary_experience_replay.extend(trend_eps_2)
        if len(trend_eps_3) > batch_size_3:
            temporary_experience_replay.extend(
                random.sample(trend_eps_3, batch_size_3))
        if len(trend_eps_3) > 0 and len(trend_eps_3) <= batch_size_3:
            temporary_experience_replay.extend(trend_eps_3)
        if len(trend_eps_4) > batch_size_4:
            temporary_experience_replay.extend(
                random.sample(trend_eps_4, batch_size_4))
        if len(trend_eps_4) > 0 and len(trend_eps_4) <= batch_size_4:
            temporary_experience_replay.extend(trend_eps_4)
        if len(trend_eps_5) > batch_size_5:
            temporary_experience_replay.extend(
                random.sample(trend_eps_5, batch_size_5))
        if len(trend_eps_5) > 0 and len(trend_eps_5) <= batch_size_5:
            temporary_experience_replay.extend(trend_eps_5)
        if len(trend_eps_6) > batch_size_6:
            temporary_experience_replay.extend(
                random.sample(trend_eps_6, batch_size_6))
        if len(trend_eps_6) > 0 and len(trend_eps_6) <= batch_size_6:
            temporary_experience_replay.extend(trend_eps_6)
        return temporary_experience_replay


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/ddqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)
agent = DQNAgent()
if agent.isTraining:
    print("Training")
    for episode in track(range(episodes)):
        rewards = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])
        env.createVoronoi()
        env.bornToDes()
        env.drawGridMap()
        while True:
            action = agent.acting(state)
            next_state, reward, done, LOD_a,LOD_b = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            # reward = -100 if done else reward
            # TODO 奖励值未同步
            (state, action, reward, next_state, done,pcc) = agent.setResemblanceValue((state, action, reward, next_state, done,  LOD_a,LOD_b))
            rewards += reward
            agent.experience_replay.append((state, action, reward, next_state, done,pcc))
            state = next_state
            if (not done) or rewards >= step_limit:  # or rewards >= step_limit
                episode_rewards.append(rewards)
                average_reward = tf.reduce_mean(episode_rewards).numpy()
                average_rewards.append(average_reward)
                max_reward = max_reward if max_reward > rewards else rewards
                agent.training()
                with summary_writer.as_default():
                    tf.summary.scalar('episode reward[s]', rewards, step=episode)
                    tf.summary.scalar('running avg reward', tf.reduce_mean(episode_rewards).numpy(), step=episode)
                break
        if (episode + 1) % 50 == 0:
            print(template.format(episode, rewards, max_reward, average_reward, agent.epsilon))
            if agent.save_model:
                agent.model.save('./model/Double_DQN/DDPG_model_sample.h5')
                print('model saved')
            if agent.save_graph:
                agent.draw(episode, episode_rewards, average_rewards, "./model/Double_DQN/DDPG_sample.png")
        if (episode + 1) % 100 == 0:
            end_time = time.time()
            print('running time: {:.2f} minutes'.format((end_time - start_time) / 60))
            print('average score in last ten episodes is: {}'.format(tf.reduce_mean(episode_rewards[-10:])))
    # env.close()

if agent.random:
    episode_rewards = []
    average_rewards = []
    max_average_reward = 0
    max_reward = 0
    for episode in range(500):
        state = env.reset()
        rewards = 0
        env.createVoronoi()
        env.bornToDes()
        env.drawGridMap()
        while True:
            # env.render()
            action = np.random.randint(action_size)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            state = next_state
            if (not done) or rewards >= step_limit:  #
                episode_rewards.append(rewards)
                average_reward = tf.reduce_mean(episode_rewards).numpy()
                average_rewards.append(average_reward)
                max_reward = max(max_reward, rewards)
                print(template.format(episode, rewards, max_reward, average_reward, "Not used"))
                break
    agent.draw(episode, episode_rewards, average_rewards, "./model/Double_DQN/DDPG_sample.png")

if agent.play:
    episode_rewards = []
    average_rewards = []
    max_average_reward = 0
    max_reward = 0
    for episode in range(500):
        state = env.reset()
        rewards = 0
        state = np.reshape(state, [1, 4])
        env.createVoronoi()
        env.bornToDes()
        env.drawGridMap()
        while True:
            # env.render()
            action = np.argmax(agent.model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            next_state = np.reshape(next_state, [1, 4])
            state = next_state
            if (not done) or rewards >= step_limit:
                episode_rewards.append(rewards)
                average_reward = tf.reduce_mean(episode_rewards).numpy()
                average_rewards.append(average_reward)
                max_reward = max(max_reward, rewards)
                # max_average_reward = max(max_average_reward,average_reward)
                print(template.format(episode, rewards, max_reward, average_reward, "Not used"))
                break
    agent.draw(episode, episode_rewards, average_rewards, "./model/Double_DQN/DDPG_sample.png")
