import gymnasium as gym
import random

env = gym.make("Blackjack-v1", render_mode="human")

state, info = env.reset()

done = False
while not done:
    action = random.choice([0, 1])

    state, reward, done, truncated, info = env.step(action)

    print(f"State: {state}, Reward: {reward}, Done: {done}")

env.close()
