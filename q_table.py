import gym
import numpy as np

"""
Unlike policy gradient methods, which attempt to learn functions which directly map an observation to an action,
Q-Learning attempts to learn the value of being in a given state, and taking a specific action there

 Bellman equation, which states that the expected long-term reward for a given action is equal to the immediate
 reward from the current action combined with the expected reward from the best future action taken
 at the following state

the Q-value for a given state (s) and action (a) should represent the current reward (r) plus
the maximum discounted (γ) future reward expected according to our own table for the next state (s’)
we would end up in. The discount variable allows us to decide how important the possible future rewards
are compared to the present reward. By updating in this way, the table slowly begins to obtain accurate
measures of the expected future reward for a given action in a given state
"""

env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
# create lists to contain total rewards and steps per episode
# jList = []
rList = []

"""
0S  1F  2F  3F
4F  5H  6F  7H
8F  9F  10F 11H
12H 13F 14F 15G
"""

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table
        # bigger of i (later of the training, the noise will have less effect of established "best" steps)
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1,:]) - Q[s, a])
        rAll += r
        s = s1
        if d == True:
            break
    # jList.append(j)
    rList.append(rAll)

print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)

"""
0S  1F  2F  3F
4F  5H  6F  7H
8F  9F  10F 11H
12H 13F 14F 15G
"""