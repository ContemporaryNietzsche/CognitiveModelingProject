import gymnasium as gym

# Initialize the Blackjack environment
env = gym.make("Blackjack-v1", render_mode="human")

# Function to get human input for the action


def get_human_action():
    while True:
        action = input("Enter 'h' to hit or 's' to stick: ").strip().lower()
        if action == 'h':
            return 1  # Hit
        elif action == 's':
            return 0  # Stick
        else:
            print("Invalid input. Please enter 'h' or 's'.")


# Start the environment
state, info = env.reset()

done = False
while not done:
    # Display the current state
    print(f"Current state: {state}")

    # Get the action from the human player
    action = get_human_action()

    # Apply the action to the environment
    state, reward, done, truncated, info = env.step(action)

    # Print the result of the action
    print(f"New state: {state}, Reward: {reward}, Done: {done}")

    # Check if the game is done
    if done:
        print("Game over!")
        if reward > 0:
            print("You win!")
        elif reward < 0:
            print("You lose!")
        else:
            print("It's a draw!")
        break

# Close the environment
env.close()
