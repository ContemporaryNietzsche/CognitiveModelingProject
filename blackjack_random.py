import gymnasium as gym
import random
import time

env = gym.make("Blackjack-v1", render_mode="human")


for i in range(10):
    state, info = env.reset()

    done = False
    while not done:
        action = random.choice([0, 1])

        state, reward, done, truncated, info = env.step(action)

        print(f"State: {state}, Reward: {reward}, Done: {done}")
        time.sleep(3)

env.close()
