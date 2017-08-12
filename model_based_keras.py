import numpy as np
import matplotlib.pyplot as plt

import gym

env = gym.make("CartPole-v0")

learning_rate = 1e-3  # Learning rate, applicable to both nn, policy and model

gamma = 0.99  # Discount factor for rewards

decay_rate = 0.99  # Decay factor for RMSProp leaky sum of grad**2

model_batch_size = 3  # Batch size used for training model nn
policy_batch_size = 3  # Batch size used for training policy nn

dimen = dimen = env.observation_space.shape[0]  # Number of dimensions in the environment


def discount(r, gamma=0.99, standardize=False):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
    """
    discounted = np.array([val * (gamma ** i) for i, val in enumerate(r)])
    if standardize:
        discounted -= np.mean(discounted)
        discounted /= np.std(discounted)
    return discounted


def step_model(sess, xs, action):
    """ Uses our trained nn model to produce a new state given a previous state and action """
    # Last state
    x = xs[-1].reshape(1, -1)

    # Append action
    x = np.hstack([x, [[action]]])

    # Predict output
    output_y = sess.run(predicted_state_m, feed_dict={input_x_m: x})

    # predicted_state_m == [state_0, state_1, state_2, state_3, reward, done]
    output_next_state = output_y[:, :4]
    output_reward = output_y[:, 4]
    output_done = output_y[:, 5]

    # First and third env outputs are limited to +/- 2.4 and +/- 0.4
    output_next_state[:, 0] = np.clip(output_next_state[:, 0], -2.4, 2.4)

    output_next_state[:, 2] = np.clip(output_next_state[:, 2], -0.4, 0.4)

    # Threshold for being done is likliehood being > 0.1
    output_done = True if output_done > 0.01 or len(xs) > 500 else False

    return output_next_state, output_reward, output_done


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

num_hidden_m = 256
dimen_m = dimen + 1

model_m = Sequential()
model_m.add(Dense(num_hidden_m, input_dim=dimen_m, activation="relu"))
model_m.add(Dense(num_hidden_m, activation="relu"))
model_m.add(Dense(dimen + 1 + 1))  # output layer: next obs, reward, gameover

model_m.compile(optimizer=Adam(lr=learning_rate), loss="mse")

# Policy network

num_hidden_p = 256
dimen_p = dimen

model_p = Sequential()
model_p.add(Dense(num_hidden_p, input_dim=dimen_p, activation="relu"))
model_p.add(Dense(2))  # Two outputs, one for action 0, one for action 1

model_p.compile(optimizer=Adam(lr=learning_rate), loss="mse")

# Keep track our our rewards
reward_sum = 0
reward_total = []

# Tracks the score on the real (non-simulated) environment to determine when to stop
episode_count = 0
num_episodes = 5000
max_num_moves = 300

# Setup array to keep track of observations, rewards and actions
observations = np.empty(0).reshape(0, dimen)
rewards = np.empty(0).reshape(0, 1)
actions = np.empty(0).reshape(0, 1)
policies = np.empty(0).reshape(0, 2)

draw_from_model = False
train_the_model = True
train_the_policy = False

num_episode = 0

observation = env.reset()

while num_episode < num_episodes:
    observation = observation.reshape(1, -1)

    # Determine the policy
    policy = model_p.predict(observation)
    policies = np.vstack([policies, policy])

    # Decide on an action based on the policy, allowing for some randomness
    action = np.argmax(model_p.predict(observation)[0])

    # Keep track of the observations and actions
    observations = np.vstack([observations, observation])
    actions = np.vstack([actions, action])

    # Determine next observation either from model or real environment

    if draw_from_model:
        output = model_m.predict(np.hstack([observation, action]))
        observation, reward, done = output[:4], output[4], output[5]
    else:
        observation, reward, done, _ = env.step(action)

    # Keep track of rewards
    reward_sum += reward
    rewards = np.vstack([rewards, reward])

    # If game is over or running long
    if done or len(observations) > max_num_moves:

        # Keep track of how many real scenarios to determine average score from real environment
        episode_count += 1

        # Keep track of rewards
        reward_total.append(reward_sum)

        # Discount rewards
        disc_rewards = discount(rewards, standardize=True)

        for idx, action, disc_reward in zip(range(len(actions)), actions, disc_rewards):
            policies[idx, int(action[0])] = disc_reward

        num_episode += 1

        observation = env.reset()

        if train_the_policy:
            model_p.train_on_batch(observations, policies)

        # Reset everything
        observations = np.empty(0).reshape(0, dimen)
        rewards = np.empty(0).reshape(0, 1)
        actions = np.empty(0).reshape(0, 1)
        policies = np.empty(0).reshape(0, 2)

        # Print periodically
        if (num_episode % (100 * policy_batch_size) == 0):
            # prob_random -= 0.1
            # prob_random = max(0.0, prob_random)
            print("Episode {} rewards: {}".format(
                num_episode, reward_sum / policy_batch_size))

        # If we our real score is good enough, quit
        if episode_count > 0:
            if (reward_sum / episode_count >= 300):
                print("Episode {} Training complete with total score of: {}".format(
                    num_episode, reward_sum / episode_count))
                break
            episode_count = 0
            reward_sum = 0

        reward_sum = 0

# See our trained bot in action

observation = env.reset()
observation
reward_sum = 0
num_move = 0

while True:
    env.render()

    x = np.reshape(observation, [1, dimen])
    y = model_p.predict(x)
    y = np.argmax(y[0])
    observation, reward, done, _ = env.step(y)
    reward_sum += reward
    num_move += 1

    if done or num_move > max_num_moves:
        print("Total score: {}".format(reward_sum))
        break