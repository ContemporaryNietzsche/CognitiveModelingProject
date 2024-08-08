import gym
env = gym.make("CartPole-v0")
env.reset()
while True:
    action = int(input("Action: "))
    if action in (0, 1):
        env.step(action)
        env.render()
