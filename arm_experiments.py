import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bandits import BernoulliBandit, ContextualBandit, GaussianBandit
from risk_sampler import RiskSampler
from solvers import Solver, EpsilonGreedy, UCB, LinUCB
from plots import *
from utils import NormalDist, AvailabilityType


def gaussian_experiment(timesteps, output_directory):
    """
    Run a small experiment on solving a Bernoulli bandit with K treatments,
    each with a randomly initialized reward probability.
    Args:
        K (int): number of treatments.
        N (int): number of time steps to try.
    """
    num_arms = 5
    reward_dists = [NormalDist(0, 1), NormalDist(1, 1), NormalDist(-1, 1), NormalDist(2, 1), NormalDist(2, 2)]
    bandit = GaussianBandit(num_arms, reward_dists)

    test_solvers = [
        EpsilonGreedy(bandit, 0.1),
        UCB(bandit)
    ]
    names = [
        r'$\epsilon$' + '-Greedy',
        "UCB",
    ]

    for solver in test_solvers:
        solver.run(timesteps)
        print(solver.estimates)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plot_filename = os.path.join(output_directory, f"results_K{num_arms}_N{timesteps}.png")
    plot_results(test_solvers, names, plot_filename)

def _get_increasing_means_dist(num_arms, variance=1):
    return [NormalDist(arm, variance) for arm in range(num_arms)]

def _get_increasing_variance_dist(num_arms, mean=0):
    return [NormalDist(mean, (0.5 * arm) + 0.5) for arm in range(num_arms)]

def arms_experiment(timesteps, output_directory):
    """
    Run a small experiment on solving a Bernoulli bandit with K treatments,
    each with a randomly initialized reward probability.
    Args:
        K (int): number of treatments.
        N (int): number of time steps to try.
    """
    availability_type = AvailabilityType.first_fraction
    num_arms = 10
    num_schemes = 4
    scheme_dists = []

    # Fix variance of arms and choose increasing means
    scheme1_dists = _get_increasing_means_dist(num_arms)
    scheme_dists.append(scheme1_dists)

    # Fix variance of arms and choose decreasing means
    scheme2_dists = _get_increasing_means_dist(num_arms)[::-1]
    scheme_dists.append(scheme2_dists)

    # Fix means of arms and choose increasing variances
    scheme3_dists = _get_increasing_variance_dist(num_arms)
    scheme_dists.append(scheme3_dists)

    # Fix means of arms and choose decreasing variances
    scheme4_dists = _get_increasing_variance_dist(num_arms)[::-1]
    scheme_dists.append(scheme4_dists)

    for idx in range(num_schemes):
        reward_dists = scheme_dists[idx]
        bandit = GaussianBandit(num_arms, reward_dists)

        test_solvers = [
           EpsilonGreedy(bandit, 0.1, availability_type),
           UCB(bandit, availability_type)
        ]
        names = [
           r'$\epsilon$' + '-Greedy',
            "UCB",
        ]

        risk_sampler = RiskSampler()
        metric_frames = []
        predicted_uncertainties = []
        for solver in test_solvers:
            metric_frame, final_uncertainties = solver.run(timesteps, risk_sampler)
            metric_frames.append(metric_frame)
            predicted_uncertainties.append(final_uncertainties)


        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        plot_filename = os.path.join(output_directory, f"results_K{num_arms}_N{timesteps}_scheme{idx}.png")
        plot_results(test_solvers, names, metric_frames, predicted_uncertainties, plot_filename)


if __name__ == '__main__':
    np.random.seed(2)
    arms_experiment(10000, "results/arms/first_fraction2")