#%%
import argparse
import datetime
import random
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from envir.GridMapEnv import GridMapEnv, Points

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', default=False)
parser.add_argument('--test', dest='test', default=False)

parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=0.1)

parser.add_argument('--train_episodes', type=int, default=200)
parser.add_argument('--test_episodes', type=int, default=10)
args = parser.parse_args()

ALG_NAME = 'DQN'
ENV_ID = 'GridMapEnv'
#%%
#GPU
# GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["SDL_VIDEODRIVER"] = "dummy"

config =  tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
sess = tf.compat.v1.Session(config=config)
#%%
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size = args.batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, next_state, done
#%%
class Agent:
    def __init__(self,env):
        self.env=env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        def create_model(input_state_shape):
            input_layer = tl.layers.Input(input_state_shape)
            layer_1 = tl.layers.Dense(n_units=64, act=tf.nn.relu)(input_layer)
            layer_2 = tl.layers.Dense(n_units=32, act=tf.nn.relu)(layer_1)
            output_layer = tl.layers.Dense(n_units=self.action_dim)(layer_2)
            return tl.models.Model(inputs=input_layer, outputs=output_layer)
        self.model = create_model([None, self.state_dim])
        self.target_model = create_model([None, self.state_dim])
        self.model.train()
        self.target_model.eval()
        self.model_optim = self.target_model_optim = tf.optimizers.Adam(lr=args.lr)

        self.epsilon = args.eps

        self.buffer = ReplayBuffer()

    def target_update(self):
        # q 值
        for weights, target_weights in zip(
                self.model.trainable_weights, self.target_model.trainable_weights):
            target_weights.assign(weights)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_value = self.model(state[np.newaxis, :])[0]
            return np.argmax(q_value)

    def replay(self):
        for _ in range(10):
            # sample an experience tuple from the dataset(buffer)
            states, actions, rewards, next_states, done = self.buffer.sample()
            # compute the target value for the sample tuple
            # targets [batch_size, action_dim]
            # Target represents the current fitting level
            target = self.target_model(states).numpy()
            # next_q_values [batch_size, action_dim]
            next_target = self.target_model(next_states)
            next_q_value = tf.reduce_max(next_target, axis=1)
            target[range(args.batch_size), actions] = rewards + (1 - done) * args.gamma * next_q_value

            # use sgd to update the network weight
            with tf.GradientTape() as tape:
                q_pred = self.model(states)
                loss = tf.losses.mean_squared_error(target, q_pred)
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.model_optim.apply_gradients(zip(grads, self.model.trainable_weights))

    def test_episode(self, test_episodes):
        for episode in range(test_episodes):
            state = self.env.reset().astype(np.float32)
            total_reward, done = 0, True
            self.env.createVoronoi()
            self.env.bornToDes()
            self.env.drawGridMap()
            while done:
                action = self.model(np.array([state], dtype=np.float32))[0]
                action = np.argmax(action)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.astype(np.float32)
                # self.env.showTrace()
                total_reward += reward
                state = next_state
                # self.env.render()
            if not done:
                self.env.reset()
            print("Test {} | episode rewards is {}".format(episode, total_reward))

    def train(self, train_episodes=200):
        if args.train:
            Dispaly_interval = 100
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = 'logs/dqn/' + current_time
            summary_writer = tf.summary.create_file_writer(log_dir)
            total_reward, done = 0, True
            total_rewards = np.empty(train_episodes)
            for episode in range(train_episodes):
                # if episode>0:
                avg_rewards = total_rewards[max(0, episode - Dispaly_interval):(episode + 1)].mean()
                # else:
                #     avg_rewards = 0
                total_rewards[episode] = total_reward
                with summary_writer.as_default():
                    tf.summary.scalar('episode reward', total_reward, step=episode)
                    tf.summary.scalar('running avg reward(100)', avg_rewards, step=episode)
                    # tf.summary.scalar('average loss)', losses, step=episode)
                total_reward, done = 0, True
                state = self.env.reset().astype(np.float32)
                self.env.createVoronoi()
                self.env.bornToDes()
                self.env.drawGridMap()
                # self.env.show()
                while done:
                    # print(state)
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.env.showTrace()
                    next_state = next_state.astype(np.float32)
                    self.buffer.push(state, action, reward, next_state, done)
                    total_reward += reward
                    state = next_state
                if not done:
                    self.env.reset()
                if len(self.buffer.buffer) > args.batch_size:
                    print(len(self.buffer.buffer))
                    self.replay()
                    self.target_update()
                print('EP{} EpisodeReward={}'.format(episode, total_reward),'\n')
                # if episode % 10 == 0:
                #     self.test_episode()
            self.saveModel()
        if args.test:
            self.loadModel()
            self.test_episode(test_episodes=args.test_episodes)

    def saveModel(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'model.hdf5'), self.model)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'target_model.hdf5'), self.target_model)
        print('Saved weights.')

    def loadModel(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        print(path)
        if os.path.exists(path):
            print('Load DQN Network parametets ...')
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'model.hdf5'), self.model)
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'target_model.hdf5'), self.target_model)
            print('Load weights!')
        else: print("No model file find, please train model first...")

#%%
if __name__ == "__main__":
    # 设置边界
    border_x = [-10, 3]
    border_y = [1, 14]
    # 奖罚,奖励>同级跳转》越级》非父子》跨页后非向下跳转》边界》层级limit
    punish = [50, 20, -10, -10, -10, -10, 3]

    born = Points(-0.87, 8.50, '8', 'r')
    dest = Points(-5.20, 6.00, '8', 'r')

    gridMap = GridMapEnv()
    # (data, points) = gridMap.readData()
    # gridMap.createVoronoi(points)
    # gridMap.bornToDes(born, dest)
    # gridMap.drawGridMap(data['x'].tolist(), data['y'].tolist(), data['color'].tolist())

    rewards=0
    agent = Agent(gridMap)
    agent.train()




# if __name__ == '__main__':
#     # 设置边界
#     border_x = [-10, 3]
#     border_y = [1, 14]
#     # 奖罚,奖励>同级跳转》越级》非父子》跨页后非向下跳转》边界》层级limit
#     punish = [50,20,-10,-30,-100,-1000,3]
#
#     born = Points(-3.47, 7.0, '', '#2471A3', 'page-info-page','0:04:01:05:02')
#     dest = Points(-5.20, 6.00, '8', 'r')
#
#     gridMap = GridMapEnv('./envir/gosper.csv',border_x,border_y,punish)
#     (data, points) = gridMap.readData()
#     gridMap.createVoronoi(points)
#     gridMap.bornToDes(born, dest)
#     gridMap.drawGridMap(data['x'].tolist(), data['y'].tolist(), data['color'].tolist())
#     # 运动,元素，页面跳转
#     rewards = 0
#     gridMap.createState()
#
#     state,reward,done,_ = gridMap.step(0)
#     gridMap.showTrace()
#     gridMap.show()
#     rewards+= reward
#     print(rewards,gridMap.observation_space)
# #
#     state,reward,done,_ = gridMap.step( 2)
#     gridMap.showTrace()
#     gridMap.show()
#     rewards += reward
#     print(rewards,gridMap.observation_space)
#
#
#     state,reward,done,_ = gridMap.action( 4)
#     gridMap.showTrace()
#     gridMap.show()
#     rewards += reward
#     print(rewards,gridMap.observation_space)
#
#
#     state,reward,done,_ = gridMap.action( 1)
#     gridMap.showTrace()
#     gridMap.show()
#     rewards += reward
#     print(rewards,gridMap.observation_space)
#
#
#     state,reward,done,_ = gridMap.action( 3)
#     gridMap.showTrace()
#     gridMap.show()
#     rewards += reward
#     print(rewards,gridMap.observation_space)
#
#
#     state,reward,done,_ = gridMap.action( 5)
#     gridMap.showTrace()
#     gridMap.show()
#     rewards += reward
#     print(rewards,gridMap.observation_space)
#
#
#     state,reward,done,_ = gridMap.action( 8)
#     gridMap.showTrace()
#     gridMap.show()
#     rewards += reward
#     print(rewards,gridMap.observation_space)
#
# #页面
#     state,reward,done,_ = gridMap.step( 6)
#     gridMap.showTrace()
#     gridMap.show()
#     rewards += reward
#     print(rewards,gridMap.observation_space)




#     # https: // blog.csdn.net / november_chopin / article / details / 107913103