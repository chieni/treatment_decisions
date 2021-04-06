import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns 

from solvers import Solver

def regret_subplot(data, ax):
    # Sub.fig. 1: Regrets in time.
    sns.lineplot(x='timestep', y='regret', hue='solver', data=data, ax=ax)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative regret')
    ax.grid('k', ls='--', alpha=0.3)

def reward_subplot(data, ax):
    # Sub.fig. 2: Cumulative rewards over time
    sns.lineplot(x='timestep', y='reward', hue='solver', data=data, ax=ax)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative reward')
    ax.grid('k', ls='--', alpha=0.3)

def mean_estimate_subplot(arm_indices, true_means, data, ax):
    # Sub.fig. 3: Probabilities estimated by solvers.
    ax.plot(arm_indices, [true_means[x] for x in arm_indices], 'k--', markersize=12, label='ground truth')
    sns.pointplot(x='arm_idx', y='mean_estimate', hue='solver', data=data, ci='sd', legend=True, ax=ax, join=False)
    ax.set_xlabel('Actions by index')
    ax.set_ylabel('Estimated mean')
    ax.grid('k', ls='--', alpha=0.3)

def std_estimate_subplot(arm_indices, true_stds, data, ax):
    # Sub.fig. 4: Estimated uncertainties
    ax.plot(arm_indices, [true_stds[x] for x in arm_indices], 'k--', markersize=12, label='ground truth std')
    sns.pointplot(x='arm_idx', y='uncertainty_estimate', hue='solver', data=data, ci='sd', legend=True, ax=ax, join=False)
    ax.set_xlabel('Actions by index')
    ax.set_ylabel('Estimated uncertainty')
    ax.grid('k', ls='--', alpha=0.3)

def action_counts_subplot(num_timesteps, data, ax):
    # Sub.fig. 5: Action counts
    data['fraction'] = data['count'] / float(num_timesteps)
    sns.lineplot(x='arm_idx', y='fraction', hue='solver', data=data, legend=True, ax=ax)
    ax.set_xlabel('Actions by index')
    ax.set_ylabel('Fraction of samples')
    ax.grid('k', ls='--', alpha=0.3)

def risk_tolerance_subplot(data, metric_frames, solver_axes, solver_names, num_arms, has_trials=False):
    # Sub.fig. 6, 7: Distribution of treatments by risk score
    for idx, solver_name in enumerate(set(solver_names)):
        metric_frame = data[data['solver'] == solver_name]
        axis = solver_axes[idx]
        metric_frame['risk_tolerance'] = pd.cut(metric_frame['risk_score'], bins=[0, 0.33, 0.66, 1], labels=['low', 'med', 'high'])
        if has_trials:
            counts_frame = metric_frame[['selected_arm', 'risk_tolerance', 'trial']].value_counts().reset_index(name='count')
        else:
            counts_frame = metric_frame[['selected_arm', 'risk_tolerance']].value_counts().reset_index(name='count')
        counts_frame['count'] = counts_frame['count'] / counts_frame['count'].sum()

        sns.lineplot(x='selected_arm', y='count', hue='risk_tolerance', data=counts_frame, ax=axis, marker='o')
        axis.set_title(solver_names[idx])
        axis.set_xlim((0, num_arms))
        axis.set_xlabel('Actions by index')
        axis.set_ylabel('Fraction of samples')
        axis.grid('k', ls='--', alpha=0.3)

def plot_results_all_trials(solver_names, metric_frames, arms_frames, plot_filename):
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

    # Concatenate metric frames for all types of solvers
    all_metric_frames = []
    for idx, metric_frame in enumerate(metric_frames):
        metric_frame['solver'] = solver_names[idx]
        all_metric_frames.append(metric_frame)
    all_metric_frame = pd.concat(all_metric_frames)

    all_arms_frames = []
    for idx, arms_frame in enumerate(arms_frames):
        arms_frame['solver'] = solver_names[idx]
        all_arms_frames.append(arms_frame)
    all_arms_frame = pd.concat(all_arms_frames)

    num_arms = all_arms_frame['arm_idx'].unique().shape[0]
    num_timesteps = all_metric_frame['timestep'].max()
    true_means = arms_frames[0]['true_mean'].values
    true_stds = arms_frames[0]['true_uncertainty'].values
    sorted_indices = range(num_arms)
    solver_axes = [ax6, ax7]

    regret_subplot(all_metric_frame, ax1)
    reward_subplot(all_metric_frame, ax2)
    mean_estimate_subplot(sorted_indices, true_means, all_arms_frame, ax3)
    std_estimate_subplot(sorted_indices, true_stds, all_arms_frame, ax4)
    action_counts_subplot(num_timesteps, all_arms_frame, ax5)
    risk_tolerance_subplot(all_metric_frame, metric_frames, solver_axes, solver_names, num_arms, True)

    plt.savefig(plot_filename)
    plt.close()

def plot_results(solver_names, metric_frames, arms_frames, plot_filename):
    """
    Plot the results by multi-armed bandit solvers.
    Args:
        solver_names (list<str)
        plot_filename (str)
    """

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

    # Concatenate metric frames for all types of solvers
    all_metric_frames = []
    for idx, metric_frame in enumerate(metric_frames):
        metric_frame['solver'] = solver_names[idx]
        all_metric_frames.append(metric_frame)
    all_metric_frame = pd.concat(all_metric_frames)

    all_arms_frames = []
    for idx, arms_frame in enumerate(arms_frames):
        arms_frame['solver'] = solver_names[idx]
        all_arms_frames.append(arms_frame)
    all_arms_frame = pd.concat(all_arms_frames)

    num_arms = all_arms_frame['arm_idx'].unique().shape[0]
    num_timesteps = all_metric_frame['timestep'].max()
    true_means = arms_frames[0]['true_mean'].values
    true_stds = arms_frames[0]['true_uncertainty'].values
    sorted_indices = range(num_arms)
    solver_axes = [ax6, ax7]

    regret_subplot(all_metric_frame, ax1)
    reward_subplot(all_metric_frame, ax2)
    mean_estimate_subplot(sorted_indices, true_means, all_arms_frame, ax3)
    std_estimate_subplot(sorted_indices, true_stds, all_arms_frame, ax4)
    action_counts_subplot(num_timesteps, all_arms_frame, ax5)
    risk_tolerance_subplot(all_metric_frame, metric_frames, solver_axes, solver_names, num_arms)

    plt.savefig(plot_filename)
    plt.close()

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