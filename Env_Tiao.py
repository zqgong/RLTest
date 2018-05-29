
import subprocess
import time
import numpy as np

import cv2
import random
# from keras.layers import *
# from keras.models import Model,load_model,Sequential
# from keras.callbacks import *


class Env:
    def __init__(self, act=100, state_size=(200, 120, 1)):
        # self.restart_btn_img = cv2.imread('./restart_button.png', cv2.IMREAD_GRAYSCALE)
        self.action_space = act
        self.dist_space = np.linspace(300, 1100, self.action_space)
        self.state_size = state_size

    def __jump(self, distance):
        # touch_x = random.randrange(300, 900)
        # touch_y = random.randrange(300, 900)
        touch_x, touch_y = 200, 1100
        subprocess.call(".\\adb\\adb shell input swipe %d %d %d %d %d" % (touch_x, touch_y, touch_x, touch_y, distance))
        time.sleep(3)

    def __capture_img(self, filename):
        subprocess.call('.\\adb\\adb shell /system/bin/screencap -p /sdcard/screenshot.png')
        time.sleep(0.1)
        subprocess.call('.\\adb\\adb pull /sdcard/screenshot.png %s' % filename)
        time.sleep(0.1)

    def __checkCrash(self, im):
        arrim = np.array(im)
        print('# =====================================')
        print('mean ', arrim.mean())
        if arrim.mean() < 155:
            return True
        return False

    def __find_restart_btn(self, screen_shot_im):

        result = self.__checkCrash(screen_shot_im)
        if result:

            x = 550
            y = 1020
            return x, y
        else:
            return -1, -1

    def __to_state(self, screen_shot_im):
        # print(np.array(screen_shot_im).shape)
        im = np.expand_dims(cv2.resize(screen_shot_im, (120, 200)) / 255.0, -1)
        # print('state_size ', im.shape)
        return im

    def dist(self, action):
        # action = 2 * action / float(self.action_space) - 1
        # d = action * 400 + 700
        d = 300 + action * 800 / (self.action_space-1)
        if d < 300:
            d = 300
        elif d > 1100:
            d = 1100

        return d

    def reset(self):
        self.__capture_img('./tmp/screenshot.png')
        im = cv2.imread('./tmp/screenshot.png', cv2.IMREAD_GRAYSCALE)
        # im = Image.open('./tmp/screenshot.png')
        btn_x, btn_y = self.__find_restart_btn(im)

        # The game has not ended yet
        if btn_x == -1:
            # Kill self
            self.__jump(1500)
            self.__capture_img('./tmp/screenshot.png')
            im = cv2.imread('./tmp/screenshot.png', cv2.IMREAD_GRAYSCALE)
            # im = Image.open('./tmp/screenshot.png')
            btn_x, btn_y = self.__find_restart_btn(im)

            assert btn_x != -1

        subprocess.call('.\\adb\\adb shell input tap %d %d' % (btn_x, btn_y))
        time.sleep(1)

        self.__capture_img('./tmp/screenshot.png')
        im = cv2.imread('./tmp/screenshot.png', cv2.IMREAD_GRAYSCALE)
        # im = Image.open('./tmp/screenshot.png')
        btn_x, btn_y = self.__find_restart_btn(im)

        assert btn_x == -1

        return self.__to_state(im)

    def step(self, action):
        '''
        action: touch time(milliseconds)
        return:
        '''

        dist = self.dist(action)
        self.__jump(dist)

        # time.sleep(1.)
        self.__capture_img('./tmp/screenshot.png')

        im = cv2.imread('./tmp/screenshot.png', cv2.IMREAD_GRAYSCALE)
        # im = Image.open('./tmp/screenshot.png')
        btn_x, btn_y = self.__find_restart_btn(im)

        # info = True
        # The game has not ended yet
        if btn_x == -1:
            state = self.__to_state(im)
            reward = 1
            done = False
            info = False
            return state, reward, done, info
        else:
            state = np.zeros(self.state_size)
            reward = -1
            done = True
            info = True
            return state, reward, done, info

# print(np.linspace(300, 1100, 21))
# print(random.randrange(21))

# a = np.random.standard_normal(size=(3,3))
# print(a)
# b = np.argmax(a, axis=1)
# print(b)
