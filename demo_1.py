import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
import tensorflow as tf
from keras import backend as K
from skimage.color import rgb2gray
from skimage.transform import resize

EPISODES = 4000


class Agent:
    def __init__(self, state_size, action_size, is_DQN=False):
        # Cartpole
        self.render = True

        self.isDQN = is_DQN
        self.state_size = state_size
        self.action_size = action_size
        self.update_target_rate = 10000

        # Cartpole DQN Hyper parameter
        # deque replay memory
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.no_op_steps = 30
        self.epsilon_min = 0.0005
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 200000
        self.batch_size = 32
        self.train_start = 10000
        self.memory = deque(maxlen=100000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        # self.sess = tf.InteractiveSession()
        # K.set_session(self.sess)
        # self.sess.run(tf.global_variables_initializer())

        self.avg_q_max, self.avg_loss = 0, 0

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

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # <s,a,r,s'> replay_memory
    def replay_memory(self, state, action, reward, next_state, done):
        if action == 2:
            action = 1
        self.memory.append((state, action, reward, next_state, done))
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon -= self.epsilon_decay
        # print(len(self.memory))

    # def train_replay(self):
    #     if len(self.memory) < self.train_start:
    #         return
    #     batch_size = min(self.batch_size, len(self.memory))
    #     mini_batch = random.sample(self.memory, batch_size)
    #
    #     # update_input = np.zeros(tuple(list(batch_size) + list(self.action_size)))
    #     # update_target = np.zeros(tuple(list(batch_size) + list(self.action_size)))
    #
    #     update_input = np.zeros((self.batch_size, self.state_size[0],
    #                              self.state_size[1], self.state_size[2]))
    #
    #     update_target = np.zeros((self.batch_size, self.action_size))
    #
    #     for i in range(batch_size):
    #         state, action, reward, next_state, done = mini_batch[i]
    #         target = self.model.predict(state)[0]
    #
    #         if done:
    #             target[action] = reward
    #         else:
    #             state = np.reshape(state, newshape=(1, 210, 160, 3))
    #             next_state = np.reshape(next_state, newshape=(1, 210, 160, 3))
    #             # print('training')
    #             if not self.isDQN:
    #
    #                 target[action] = reward + self.discount_factor * \
    #                              self.target_model.predict(next_state)[0][np.argmax(target)]
    #             else:
    #                 target[action] = reward + self.discount_factor * \
    #                                       np.amax(self.target_model.predict(next_state)[0])
    #         update_input[i] = state
    #         update_target[i] = target
    #
    #     self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size, ))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        value = self.model.predict(next_history)
        target_value = self.target_model.predict(next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                target[i] = reward[i] + self.discount_factor * \
                                        target_value[i][np.argmax(value[i])]

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    env = gym.make('PongNoFrameskip-v4')
    # state_size = env.observation_space.shape
    state_size = (84, 84, 8)
    action_size = env.action_space.n
    print(state_size, action_size)

    agent = Agent(state_size, action_size)
    scores, episodes, global_step = [], [], 0

    for i_episode in range(EPISODES):
        observation_ = env.reset()
        done = False
        score = 0
        state = env.reset()
        # next_state = state

        # print(state.shape, state_tmp.shape)
        # state = np.reshape(state, [1, state_size])
        # print('episode ', i_episode)

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, agent.no_op_steps)):
            state, _, _, _ = env.step(1)
        # fake_action = 0
        state = pre_processing(state)
        history = np.stack((state, state, state, state, state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 8))

        action_count = 0

        while not done:
            if agent.render:
                env.render()

            global_step += 1
            action_count += 1

            # state = np.reshape(state, newshape=(1, 210, 160, 3))
            action = agent.get_action(history)
            # state, reward, done, info = env.step(action)

            # action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            next_state = pre_processing(next_state)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :7], axis=3)

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(history, action, reward, next_history, done)
            # every some time interval, train model
            # if action_count % 100 == 0:
            agent.train_replay()
            score += reward

            # if global_step % agent.update_target_rate == 0:
            #     print('update target model')
            #     agent.update_target_model()

            if done:
                env.reset()
                agent.update_target_model()
                print("episode:", i_episode, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step)
                break
            else:
                state = next_state
