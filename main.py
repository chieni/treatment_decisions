import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bandits import BernoulliBandit, ContextualBandit, GaussianBandit
from solvers import Solver, EpsilonGreedy, UCB, LinUCB
from plots import *
from utils import NormalDist


def make_design_matrix(n_trial, n_arms, n_feature):
    '''
    Feature vector is iid and simulated according to Uniform[0, 1]
    '''
    available_arms = np.arange(n_arms)
    X = np.array([[np.random.uniform(low=0, high=1, size=n_feature) for _ in available_arms] for _ in np.arange(n_trial)])
    return X

def make_theta(n_arms, n_feature, best_arms, bias = 1):
    '''
    Simulated accorindg to N(O_d, cI_d), c = 0.25
    High performing arms (best_arms) are created by adding positive bias to theta.
    '''
    true_theta = np.array([np.random.normal(size=n_feature, scale=0.25) for _ in np.arange(n_arms)])
    true_theta[best_arms] = true_theta[best_arms] + bias
    return true_theta

def make_regret(payoff, oracle):
    return np.cumsum(oracle - payoff)

def experiment(num_arms, timesteps):
    """
    Run a small experiment on solving a Bernoulli bandit with K treatments,
    each with a randomly initialized reward probability.
    Args:
        K (int): number of treatments.
        N (int): number of time steps to try.
    """
    bandit = BernoulliBandit(num_arms)
    print("Randomly generated Bernoulli bandit has reward probabilities:\n", bandit.reward_probs)
    print("The best treatment has index: {} and proba: {}".format(
        max(range(num_arms), key=lambda i: bandit.reward_probs[i]), max(bandit.reward_probs))
    )

    test_solvers = [
        EpsilonGreedy(bandit, 0.1),
        UCB(bandit),
       # LinUCB(bandit)
    ]
    names = [
        r'$\epsilon$' + '-Greedy',
        "UCB",
        #"LinUCB"
    ]

    for solver in test_solvers:
        solver.run(timesteps)
        print(solver.estimates)

    plot_results(test_solvers, names, "results_K{}_N{}.png".format(num_arms, timesteps))

def context_experiment(num_arms, num_timesteps):
    """
    Run a small experiment on solving a Bernoulli bandit with K treatments,
    each with a randomly initialized reward probability.
    Args:
        K (int): number of treatments.
        N (int): number of time steps to try.

    Creates 4 plots:
    - true_values_plot.png: Visualisation of true theta values with simulated bandit run.
    - regret_plot.png: Visualisation of regret over time for bandit training.
    - estimates_plot.png: Visualition of difference between true theta and estimated theta for each treatment arm,
                          over time - meant to capture convergence.
    - theta_plot.png: Visaulisation of true theta and estimated theta for each dimension of theta.
    """
    num_features = 5
    BEST_ARMS = [3, 7, 9, 15]
    alpha = 1

    # Simulation of design matrix and weight vector
    true_theta = make_theta(num_arms, num_features, BEST_ARMS, bias=1)
    context = make_design_matrix(num_timesteps, num_arms, num_features)
    bandit = ContextualBandit(num_arms, context=context, true_theta=true_theta)
    
    rewards_dist = np.array([[bandit.select_arm(arm, context[t, arm]) for arm in np.arange(num_arms)] for t in np.arange(num_timesteps)])
    ave_reward = np.mean(rewards_dist, axis=0)
    overall_ave_reward = np.mean(ave_reward)

    plot_true_values(num_arms, ave_reward, true_theta)

    solver = LinUCB(bandit, alpha=alpha)
    theta, predictions, selected_arms, rewards, oracle, random_rewards = solver.run(num_timesteps)
    
    # Plot theta estimates (in comparison to true theta)
    plot_estimates(theta, true_theta, num_arms, num_features, abs_ylim=1.5, ncol=4)
    
    # Plot regret
    regret = make_regret(rewards, oracle)
    plt.plot(regret, label=f"alpha={alpha}")
    random_regret = make_regret(random_rewards, oracle)
    plt.plot(random_regret, label="random", linestyle='--')
    plt.legend()
    plt.savefig("regret_plot.png", dpi=300)
    plt.close()

    # Plot estimated theta values in comparison to true theta, for each dimension
    fig = plt.figure(figsize=(4 * num_features, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)
    final_theta = theta[-1, :, :]
    for feature_idx in range(num_features):
        ax = fig.add_subplot(100 + (num_features * 10) + (feature_idx + 1))
        ax.plot(range(bandit.num_arms), true_theta[:, feature_idx], 'k--', markersize=12)
        ax.plot(range(bandit.num_arms), final_theta[:, feature_idx], 'x', markeredgewidth=2)
        ax.set_xlabel('Treatment arm')
        ax.set_ylabel('Theta')
        ax.set_xticks(range(num_arms))
    
    fig.tight_layout()
    fig.savefig("theta_plot.png", dpi=300)

if __name__ == '__main__':
    context_experiment(5, 200)