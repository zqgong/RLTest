
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import random
import numpy as np
from collections import deque
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, AveragePooling2D
from keras.optimizers import Adam, RMSprop
from keras.losses import categorical_crossentropy
from keras.models import Sequential
import tensorflow as tf
import pickle
import copy
from keras import backend as K

import time

from Env_Tiao import Env


class Agent:
    def __init__(self, state_size, action_size, is_DQN=False, load_model=False):

        self.render = True

        self.isDQN = is_DQN
        self.state_size = state_size
        self.action_size = action_size
        self.update_target_rate = 10000

        # DQN 
        self.discount_factor = 0.99
        self.learning_rate = 0.0005
        self.epsilon = 1.0
        self.no_op_steps = 20
        self.epsilon_min = 0.0005
        self.epsilon_decay = 0.0002
        self.batch_size = 64
        self.train_start = 100
        self.memory = deque(maxlen=3000)
        self.channel = 4

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()
        self.should_load_model = load_model
        self.e = 0
        if self.should_load_model:
            print('loading model')
            self.epsilon = self.epsilon_min

            self.memory = pickle.load(open('./save/dqn/dqn_mem.pkl', 'rb'))
            _, self.e, _ = pickle.load(open('./save/dqn/dqn_stat.pkl', 'rb'))
            self.model.load_weights('./save/dqn/dqn_actor.h5')
            self.update_target_model()
            print('len of memory is ', len(self.memory))

        self.avg_q_max, self.avg_loss, self.loss_ = 0, 0, 0

    def optimizer(self):
        a = K.placeholder(shape=(None, ), dtype='int32')
        y = K.placeholder(shape=(None, ), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)

        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.0003, epsilon=0.0001)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    def _leaky_relu(self, x, leakiness=0.1):
        return tf.where(tf.less(x, 0.0), leakiness * x, x)

    def build_model(self):
        model = Sequential()
        # model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), input_shape=self.state_size))
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=(self.state_size[0], self.state_size[1], self.channel)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(self.action_size))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            is_random = True
            return random.randrange(self.action_size), is_random
        else:
            is_random = False
            # state = np.expand_dims(state, 0)
            q_value = self.model.predict(state)
            return np.argmax(q_value[0]), is_random

    # <s,a,r,s'> 
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        # print(self.epsilon)
        batch = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch)
        # print(mini_batch)
        history = np.zeros((batch, self.state_size[0],
                            self.state_size[1], self.channel))
        next_history = np.zeros((batch, self.state_size[0],
                                 self.state_size[1], self.channel))
        target = np.zeros((batch, ))
        action, reward, dead = [], [], []

        for i in range(batch):
            history[i] = np.float32(mini_batch[i][0][0])
            next_history[i] = np.float32(mini_batch[i][3][0])
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        value = self.model.predict(next_history)
        target_value = self.target_model.predict(next_history)

        for i in range(batch):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        target_value[i][np.argmax(value[i])]

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]
        self.loss_ = loss[0]

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':

    episodes = 10000

    # 为agent初始化gym环境参数
    env = Env(act=17)
    state_size = env.state_size
    action_size = env.action_space
    # agent = Agent()
    agent = Agent(state_size, action_size, load_model=True)
    # agent.actor.model.summary()
    # agent.critic.model.summary()

    scores = []

    e = agent.e
    print('starting episode, ', e)
    global_step = 0

    # 游戏的主循环
    while e < episodes:
        steps = 0
        try:
            # 在每次游戏开始时复位状态参数
            state = env.reset()
            history = np.stack(tuple([state]*agent.channel), axis=2)
            history = np.reshape([history], (1, state_size[0], state_size[1], agent.channel))

            for time_t in range(10000):

                # 选择行为
                action, is_random = agent.get_action(history)

                if is_random:
                    print('jump dist:', env.dist(action), 'which is random action')
                else:
                    print('jump dist:', env.dist(action), 'which is NOT random action')

                steps += 1
                global_step += 1
                
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape([next_state], (1, state_size[0], state_size[1], 1))
                next_history = np.append(next_state, history[:, :, :, :agent.channel-1], axis=3)

                # 记忆先前的状态，行为，回报与下一个状态
                agent.replay_memory(history, action, reward, next_history, done)

                agent.train_replay()
                # print(done)

                if (global_step + 1) % agent.no_op_steps == 0 and len(agent.memory) > agent.train_start:
                    print('update target model......')
                    agent.update_target_model()

                # 如果游戏结束done被置为ture
                # 除非agent没有完成目标
                if done:

                    # 打印分数并且跳出游戏循环
                    print("episode: {}/{}, score: {}"
                          .format(e, episodes, time_t))
                    scores.append(time_t)
                    break

                # 使下一个状态成为下一帧的新状态
                history = copy.deepcopy(next_history)

                # 保存模型
                if steps % 100 == 0 and steps != 0:
                    print('save model v1')
                    pickle.dump((scores, e, steps), open('./tmp/dqn_stat.pkl', 'wb'))
                    pickle.dump(agent.memory, open('./tmp/dqn_mem.pkl', 'wb'))

                    agent.save_model('./tmp/dqn_actor.h5')

                elif global_step % 500 == 0:
                    print('save model v2')
                    pickle.dump((scores, e, steps), open('./tmp/dqn_stat.pkl', 'wb'))
                    pickle.dump(agent.memory, open('./tmp/dqn_mem.pkl', 'wb'))

                    agent.save_model('./tmp/dqn_actor.h5')

            print('episode', e, 'step(global step):', steps, '/', global_step, 'epsilon:', agent.epsilon, 'loss:', agent.loss_, 'memory:', len(agent.memory))
            
            e += 1

        except:
            time.sleep(3)

