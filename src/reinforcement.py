import numpy as np
import matplotlib.pyplot as plt

def simulate_study_strategy():
    study_options = np.arange(1, 11)
    rewards = [5 * h + np.random.normal(0, 10) for h in study_options]
    best_hours = study_options[np.argmax(rewards)]

    plt.figure(figsize=(6,4))
    plt.plot(study_options, rewards, marker='o')
    plt.xlabel("Study Hours")
    plt.ylabel("Reward")
    plt.title("RL Simulation: Optimal Study Hours")
    plt.grid(True)
    plt.savefig("outputs/rl_rewards.png")
    plt.show()
    return best_hours
