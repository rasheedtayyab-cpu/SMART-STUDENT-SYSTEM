import numpy as np
import matplotlib.pyplot as plt

def find_optimal_study_hours():
    study_hours_options = np.arange(1, 11)
    rewards = [5 * hours + np.random.normal(0, 10) for hours in study_hours_options]
    best_hours = study_hours_options[np.argmax(rewards)]

    plt.figure(figsize=(6,4))
    plt.plot(study_hours_options, rewards, marker='o')
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Reward")
    plt.title("Reinforcement Learning: Optimal Study Hours")
    plt.grid(True)
    plt.savefig("outputs/optimal_study_hours.png")
    plt.show()
    
    return best_hours
