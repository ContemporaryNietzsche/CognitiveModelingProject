# CartPole game that can take player input

import gym
import time
import sys
import termios
import tty


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


env = gym.make('CartPole-v1')
env.reset()

print("Use 'a' to move left, 'd' to move right, and 'q' to quit.")

while True:
    env.render()
    action = None

    key = getch()
    if key == 'a':
        action = 0  # Push cart to the left
    elif key == 'd':
        action = 1  # Push cart to the right
    elif key == 'q':
        print("Exiting...")
        break

    if action is not None:
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished")
            env.reset()

env.close()
