import numpy as np
import matplotlib.pyplot as plt

def simulate_study_strategy():
    study_options = np.arange(1, 11)
    rewards = []

    for hours in study_options:
        # Randomized reward
        reward = 5 * hours + np.random.normal(0, 10)
        rewards.append(reward)

    best_hours = study_options[np.argmax(rewards)]

    plt.figure(figsize=(6,4))
    plt.plot(study_options, rewards, marker='o')
    plt.xlabel("Study Hours")
    plt.ylabel("Reward (Score Improvement)")
    plt.title("RL Simulation: Optimal Study Hours")
    plt.grid(True)
    plt.savefig("outputs/rl_rewards.png")
    plt.show()

    return best_hours
