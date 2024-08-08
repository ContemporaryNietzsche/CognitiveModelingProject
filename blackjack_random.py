import gymnasium as gym
import random

# Initialize the Blackjack environment
env = gym.make("Blackjack-v1", render_mode="human")

# Start the environment
state, info = env.reset()

done = False
while not done:
    # Take a random action: 0 (stick) or 1 (hit)
    action = random.choice([0, 1])

    # Apply the action to the environment
    state, reward, done, truncated, info = env.step(action)

    # Print the state and reward
    print(f"State: {state}, Reward: {reward}, Done: {done}")

# Close the environment
env.close()
