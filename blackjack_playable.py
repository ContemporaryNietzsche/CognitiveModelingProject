import gymnasium as gym

env = gym.make("Blackjack-v1", render_mode="human")


def get_human_action():
    while True:
        action = input("Enter 'h' to hit or 's' to stick: ").strip().lower()
        if action == 'h':
            return 1  # Hit
        elif action == 's':
            return 0  # Stick
        else:
            print("Invalid input. Please enter 'h' or 's'.")


state, info = env.reset()

done = False
while not done:
    print(f"Current state: {state}")

    action = get_human_action()

    state, reward, done, truncated, info = env.step(action)

    print(f"New state: {state}, Reward: {reward}, Done: {done}")

    if done:
        print("Game over!")
        if reward > 0:
            print("You win!")
        elif reward < 0:
            print("You lose!")
        else:
            print("It's a draw!")
        break

env.close()
