#!/usr/bin/env python2
from __future__ import print_function

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
# from env_ir import RLBot
from robobo import SimulationRobobo
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation

def train():
    env = SimulationRobobo()
    env.connect(address='192.168.1.135', port=19997)

    try:
        model = load_model('model_ir.hdf5')
    except:
        model = Sequential()
        model.add(Dense(units=10, input_dim=8))
        model.add(Activation("relu"))
        model.add(Dense(units=6))
        model.add(Activation("relu"))
        model.add(Dense(units=3))
        model.add(Activation("relu"))
        model.compile(optimizer='Adam', loss='mse')

    # Set learning parameters
    y = .99
    e = 0.1
    num_episodes = 1500
    num_steps = 300
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    lList = []
    SPEED = 100

    for i in range(num_episodes):
        # Reset environment and get first new observation
        env.reset()

        observations, reward = env.move(SPEED, SPEED)
        obervations = observations['ir_sensor'].reshape((1, -1))
        reward = reward['ir_sensor']

        Q = model.predict(observations)
        action = Q.argmax()
        rAll = 0
        done = False
        loss = 0
        
        # The Q-Network
        for j in range(num_steps):
            print("Step {} | State: {} | Action: {} | Reward: {}".format(j, observations, action, reward))
            # Choose an action by greedily (with e chance of random action)
            # from the Q-network
            Q = model.predict(observations)
            action = Q.argmax()

            if np.random.rand(1) < e:
                action = np.random.randint(3)
                print("e = {}. Choosing Random Action: {}".format(e, action))

            # Get new state and reward from environment
            speed = np.zeros(2)

            # Q -> left, right, forward, break
            if action == 0:
                left = 0
                right = SPEED
            if action == 1:
                left = SPEED
                right = 0
            if action == 2:
                left = SPEED
                right = SPEED
            if action == 3:
                left = 0
                right = 0

            s_observations, r_reward = env.move(left, right)
            s_observations = s_observations['ir_sensor'].reshape((1, -1))
            r_reward = r_reward['ir_sensor']

            # Obtain the Q' values by feeding the new state through our network
            Q_ = model.predict(s_observations)

            # Obtain maxQ' and set our target value for chosen action.
            maxQ_ = np.max(Q_)
            targetQ = Q
            targetQ[0, action] = reward + y * maxQ_

            # Train our network using target and predicted Q values
            loss += model.train_on_batch(observations, targetQ)
            rAll += reward
            observations = s_observations
            reward = r_reward
            if done is True:
                break

        # Reduce chance of random action as we train the model.
        e -= 0.001
        jList.append(j)
        rList.append(rAll)
        lList.append(loss)
        print("Episode: " + str(i))
        print("Loss: " + str(loss))
        print("e: " + str(e))
        print("Reward: " + str(rAll))
        pickle.dump({'jList': jList, 'rList': rList, 'lList': lList},
                    open("history_ir.p", "wb"))
        model.save('model_ir.hdf5')

    print("Average loss: " + str(sum(lList) / num_episodes))
    print("Average number of steps: " + str(sum(jList) / num_episodes))
    print("Average reward: " + str(sum(rList) / num_episodes))

    plt.plot(rList)
    plt.plot(jList)
    plt.plot(lList)

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print('Exiting.')
