#import gym
#import pygame
#from gym.utils.play import play
#mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
#play(gym.make("CartPole-v0"), keys_to_action=mapping)

#env = gym.make("CartPole-v0")
#env.reset()
#while True:
#    action = int(input("Action: "))
#    if action in (0, 1):
#        env.step(action)
#        env.render()
import gymnasium as gym
import pygame
import numpy as np
import sys

# Initialize pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption('CartPole - Use Left/Right Arrow Keys')

# Set up the environment
env = gym.make('CartPole-v1', render_mode="rgb_array")
env.reset()

# Key mapping
action_mapping = {
    pygame.K_LEFT: 0,  # Push cart to the left
    pygame.K_RIGHT: 1  # Push cart to the right
}

clock = pygame.time.Clock()
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break

    keys = pygame.key.get_pressed()
    action = None
    if keys[pygame.K_LEFT]:
        action = 0
    elif keys[pygame.K_RIGHT]:
        action = 1

    if action is not None:
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Display the rendered environment
        screen.fill((255, 255, 255))
        img = pygame.surfarray.make_surface(np.transpose(env.render(), axes=(1, 0, 2)))
        screen.blit(img, (0, 0))
        pygame.display.flip()

        if done:
            env.reset()

    clock.tick(30)  # Limit the frame rate to 30 FPS

env.close()
pygame.quit()
