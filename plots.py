import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns 

from solvers import Solver


def plot_results(solvers, solver_names, metric_frames, predicted_uncertainties, figname):
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

    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4)
    # fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ax5 = plt.subplot(gs[4])
    ax6 = plt.subplot(gs[5])
    ax7 = plt.subplot(gs[6])

    # ax1 = fig.add_subplot(151)
    # ax2 = fig.add_subplot(152)
    # ax3 = fig.add_subplot(153)
    # ax4 = fig.add_subplot(154)
    # ax5 = fig.add_subplot(155)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend()
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Cumulative rewards over time
    all_metric_frames = []
    for idx, metric_frame in enumerate(metric_frames):
        metric_frame['solver'] = solver_names[idx]
        all_metric_frames.append(metric_frame)
    all_metric_frame = pd.concat(all_metric_frames)
    sns.lineplot(x='timestep', y='reward', hue='solver', data=all_metric_frame, ax=ax2)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Cumulative reward')
    ax2.grid('k', ls='--', alpha=0.3)


    # Sub.fig. 3: Probabilities estimated by solvers.
    #sorted_indices = sorted(range(b.num_arms), key=lambda x: b.reward_probs[x])
    sorted_indices = range(b.num_arms)
    ax3.plot(range(b.num_arms), [b.reward_probs[x] for x in sorted_indices], 'k--', markersize=12, label='ground truth')
    for idx, s in enumerate(solvers):
        ax3.plot(range(b.num_arms), [s.estimates[x] for x in sorted_indices], 'x', markeredgewidth=2, label=solver_names[idx])
    ax3.legend()
    ax3.set_xlabel('Actions by index')
    ax3.set_ylabel('Estimated mean')
    ax3.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 4: Estimated uncertainties
    if b.reward_variances is not None:
        ax4.plot(range(b.num_arms), [np.sqrt(b.reward_variances[x]) for x in sorted_indices], 'k--', markersize=12, label='ground truth std')
    for idx, s in enumerate(solvers):
        ax4.plot(range(b.num_arms), [predicted_uncertainties[idx][x] for x in sorted_indices], 'x', markeredgewidth=2, label=solver_names[idx])
    ax4.legend()
    ax4.set_xlabel('Actions by index')
    ax4.set_ylabel('Estimated uncertainty')
    ax4.grid('k', ls='--', alpha=0.3)


    # Sub.fig. 5: Action counts
    for idx, s in enumerate(solvers):
        ax5.plot(range(b.num_arms), np.array(s.counts) / float(len(s.regrets)), label=solver_names[idx])
    ax5.legend()
    ax5.set_xlabel('Actions by index')
    ax5.set_ylabel('Frac. # trials')
    ax5.grid('k', ls='--', alpha=0.3)


    # Sub.fig. 6, 7: Distribution of treatments by risk score
    solver_axes = [ax6, ax7]
    for idx, metric_frame in enumerate(metric_frames):
        axis = solver_axes[idx]
        metric_frame['risk_tolerance'] = pd.cut(metric_frame['risk_score'], bins=[0, 0.33, 0.66, 1], labels=['low', 'med', 'high'])
        counts_frame = metric_frame[['selected_arm', 'risk_tolerance']].value_counts().reset_index(name='count')
        counts_frame['count'] = counts_frame['count'] / counts_frame['count'].sum()

        sns.lineplot(x='selected_arm', y='count', hue='risk_tolerance', data=counts_frame, ax=axis, marker='o')
        axis.set_title(solver_names[idx])
        axis.set_xlim((0, b.num_arms))
        axis.set_xlabel('Actions by index')
        axis.set_ylabel('Fraction of samples')
        axis.grid('k', ls='--', alpha=0.3)

    plt.savefig(figname)

def plot_estimates(theta, true_theta, num_arms, num_features, abs_ylim = None, ncol = 4):
    '''
    Plot theta estimates by timepoint for each arm. Plots the difference between
    theta and true_theta if the true_theta parmeters is provided. This meant to to 
    illustrate convergence in training.

    x-axis: difference in theta and true_theta value
    y-axis: timepoints
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
    Visulation of the true theta values and ave_reward, average reward values per arm, with the 
    bandit run through a simulation of num_timesteps.
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
    Lineplot of regret by timesteps for different alpha values. Also includes
    random regret, which is when a bandit arm is selected by random (instead of the arm with
    the highest confidence bound, as for LinUCB)
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