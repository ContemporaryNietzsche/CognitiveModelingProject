import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, state_dim, action_dim, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(lambda: np.zeros(action_dim))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.q_table[state]
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state]) if not done else 0
        new_value = (1 - self.lr) * old_value + self.lr * \
            (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = gym.make("Blackjack-v1", render_mode="human")

state_dim = 3
action_dim = env.action_space.n
agent = QLearningAgent(state_dim, action_dim)

num_episodes = 100
net_wins = []

cumulative_wins = 0

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        state = tuple(state)
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = tuple(next_state)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if total_reward > 0:
        cumulative_wins += 1
    elif total_reward < 0:
        cumulative_wins -= 1

    net_wins.append(cumulative_wins)

    if episode % 10 == 0:
        print(f"Episode {episode} completed, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

env.close()

# Plotting the graph
plt.plot(range(num_episodes), net_wins, label='Net Wins')
plt.xlabel('Number of Games')
plt.ylabel('Net Wins')
plt.title('Net Wins (Cumulative Wins - Losses) over Time')
plt.legend()
plt.show()
