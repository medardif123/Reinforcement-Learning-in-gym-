import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.1
gamma = 0.6
epsilon = 0.1
#epsilon greedy function
def get_action(state):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    return action

# For plotting metrics
training_rewards = []
training_penalties = []

episodes = 10001

for i in range(1,episodes):
    state = env.reset()

    epochs = 0
    penalties = 0
    rewards = 0
    done = False

    while not done:
        action = get_action(state)
        next_state, reward, done, info = env.step(action)

        q_table[state, action] = q_table[state, action] +alpha*(reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        rewards += reward
    training_rewards.append(rewards)
    training_penalties.append(penalties)
    if i % 1000 == 0:
        print(f"Episode: {i} reward : {training_rewards[-1]} penalties : {training_penalties[-1]}")

plt.plot(training_rewards)
plt.xlabel('episodes')
plt.ylabel('reward')
plt.show()
