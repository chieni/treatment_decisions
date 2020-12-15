import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bandits import BernoulliBandit, ContextualBandit
from solvers import Solver, EpsilonGreedy, UCB, LinUCB

def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.
    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Probabilities estimated by solvers.
    #sorted_indices = sorted(range(b.num_arms), key=lambda x: b.reward_probs[x])
    sorted_indices = range(b.num_arms)
    ax2.plot(range(b.num_arms), [b.reward_probs[x] for x in sorted_indices], 'k--', markersize=12)
    for s in solvers:
        ax2.plot(range(b.num_arms), [s.estimates[x] for x in sorted_indices], 'x', markeredgewidth=2)
    ax2.set_xlabel('Actions sorted by ' + r'$\theta$')
    ax2.set_ylabel('Estimated')
    ax2.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 3: Action counts
    for s in solvers:
        ax3.plot(range(b.num_arms), np.array(s.counts) / float(len(solvers[0].regrets)), ls='dashed', lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)

    plt.savefig(figname)

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
    bandit = BernoulliBandit(num_arms, context=context)
    print("Randomly generated Bernoulli bandit has reward probabilities:\n", bandit.reward_probs)
    print("The best treatment has index: {} and proba: {}".format(
        max(range(num_arms), key=lambda i: bandit.reward_probs[i]), max(bandit.reward_probs))
    )

    test_solvers = [
        # EpsilonGreedy(bandit, 0.1),
        # UCB(bandit),
        LinUCB(bandit)
    ]
    names = [
        # r'$\epsilon$' + '-Greedy',
        # "UCB",
        "LinUCB"
    ]

    for solver in test_solvers:
        solver.run(timesteps)
        print(solver.estimates)

    plot_results(test_solvers, names, "results_K{}_N{}.png".format(num_arms, timesteps))

def plot_estimates(theta, true_theta, num_arms, num_features, abs_ylim = None, ncol = 4):
    '''
    Plot theta estimates by timepoint for each arm
    '''
    BEST_ARMS = [3, 7, 9, 15]
    plt.figure(figsize=(12.5, 17.5))
    for i, arm in enumerate(np.arange(num_arms)):
        plt.subplot(np.ceil(num_arms/ncol), ncol, 1+i)
        if true_theta is not None:
            data_to_plot = pd.DataFrame(theta[:, arm, :]) - true_theta[arm]
        else:
            data_to_plot = pd.DataFrame(theta[:, arm, ])
        plt.plot(data_to_plot)
        
        if (arm in BEST_ARMS):
            title = 'Arm: ' + str(arm) + " (best)"
        else:
            title = "Arm: " + str(arm)
        plt.title(title)
        
        if abs_ylim is not None:
            plt.ylim([-abs_ylim, abs_ylim])
    plt.legend(["c"+str(feature) for feature in np.arange(num_features)])
    plt.savefig("estimates_plot.png", dpi=300)
    plt.close()

def plot_true_values(num_arms, ave_reward, true_theta):
    '''
    Visulation of 'true values'
    '''
    f, (left, right) = plt.subplots(1, 2, figsize=(15, 10))
    f.suptitle(t="Visualizing of simulated parameters: true theta and average reward", fontsize=20)
    # True theta
    left.matshow(true_theta)
    f.colorbar(left.imshow(true_theta), ax = left)
    left.set_xlabel("feature number")
    left.set_ylabel("arm number")
    left.set_yticks(np.arange(num_arms))
    left.set_title("True theta matrix")
    # Average reward
    right.bar(np.arange(num_arms), ave_reward)
    right.set_title("Average reward per arm")
    right.set_xlabel("arm number")
    right.set_ylabel("average reward")
    plt.savefig('true_values_plot.png', dpi=300)
    plt.close()

def plot_regret(bandit, num_timesteps, true_theta):
    '''
    Vis of regret by alphas
    '''
    alpha_to_test = [0, 1, 2.5, 5, 10, 20]
    for alpha in alpha_to_test:
        solver = LinUCB(bandit, alpha=alpha)
        theta, predictions, selected_arms, rewards, oracle, random_rewards = solver.run(num_timesteps, true_theta)
        regret = make_regret(rewards, oracle)
        plt.plot(regret, label=f"alpha={alpha}")

    random_regret = make_regret(random_rewards, oracle)
    plt.plot(random_regret, label="random", linestyle='--')
    plt.legend()
    plt.savefig("regret_plot.png", dpi=300)
    plt.close()

def context_experiment(num_arms, num_timesteps):
    """
    Run a small experiment on solving a Bernoulli bandit with K treatments,
    each with a randomly initialized reward probability.
    Args:
        K (int): number of treatments.
        N (int): number of time steps to try.
    """
    num_features = 5
    BEST_ARMS = [3, 7, 9, 15]

    # Simulation of design matrix and weight vector
    true_theta = make_theta(num_arms, num_features, BEST_ARMS, bias=1)
    context = make_design_matrix(num_timesteps, num_arms, num_features)
    bandit = ContextualBandit(num_arms, context=context, true_theta=true_theta)
    
    rewards_dist = np.array([[bandit.select_arm(arm, context[t, arm]) for arm in np.arange(num_arms)] for t in np.arange(num_timesteps)])
    ave_reward = np.mean(rewards_dist, axis=0)
    overall_ave_reward = np.mean(ave_reward)

    plot_true_values(num_arms, ave_reward, true_theta)

    solver = LinUCB(bandit, alpha=1)
    theta, predictions, selected_arms, rewards, oracle, random_rewards = solver.run(num_timesteps)
    plot_estimates(theta, true_theta, num_arms, num_features, abs_ylim=1.5, ncol=4)


    # Working on
    # plot the scatterplot for the different dimensions on diff plots or colors? 
    # plt.plot(range(bandit.num_arms), true_theta, 'k--', markersize=12)
    # plt.plot(range(bandit.num_arms), theta[-1, :, :], 'x', markeredgewidth=2)
    # plt.set_xlabel('Treatment arms')
    # plt.set_ylabel('Estimated ' + r'$\theta$')
    # plt.grid('k', ls='--', alpha=0.3)
    # plt.show()

if __name__ == '__main__':
    context_experiment(16, 2000)