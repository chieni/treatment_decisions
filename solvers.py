import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import NormalDist, AvailabilityType


np.random.seed(10)

class Solver:
    def __init__(self, bandit, availability_type, init_proba):
        """
        bandit (Bandit): the target bandit to solve.
        """
        self.bandit = bandit
        self.availability_type = availability_type

        self.counts = [0] * self.bandit.num_arms
        self.cumulative_reward = 0.
        self.regret = 0.  # Cumulative regret.

        # self.estimates is the empirical mean of the observed reward samples
        self.estimates = [init_proba] * self.bandit.num_arms # Optimistic initialization

    def update_regret(self, selected_arm):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_reward_prob - self.bandit.reward_probs[selected_arm]
        return self.regret
    
    def update_reward(self, reward):
        self.cumulative_reward += reward
        return self.cumulative_reward

    def availability_function(self, sample_risk_score, uncertainties):
        all_arms_indices = self.bandit.arm_indices
        num_arms = self.bandit.num_arms
        end_idx = max(1, int(np.ceil(sample_risk_score * num_arms)))

        # Select first fraction of arms based on risk score
        if self.availability_type == AvailabilityType.first_fraction:
            return all_arms_indices[:end_idx]
        
        if self.availability_type == AvailabilityType.mean_estimates:
            # Note: if everything is equal, do not sort the array, otherwise the array just flips
            if len(set(self.estimates)) == 1:
                sorted_estimates = list(all_arms_indices)
            else:
                sorted_estimates = np.argsort(self.estimates)[::-1]
            return sorted_estimates[:end_idx]

        if self.availability_type == AvailabilityType.uncertainty_estimates:
            if len(set(uncertainties)) == 1:
                sorted_uncertainties = list(all_arms_indices)
            else:
                sorted_uncertainties = np.argsort(uncertainties)
            return sorted_uncertainties[:end_idx]
            
        return all_arms_indices
    
    def get_uncertainty(self, arm, timestep):
        return np.sqrt(2. * np.log(1 + timestep * np.log(timestep)**2) / (self.counts[arm])) 

    def update_estimates(self, selected_arm):
        reward = self.bandit.select_arm(selected_arm)
        self.estimates[selected_arm] += 1. / (self.counts[selected_arm]) * (reward - self.estimates[selected_arm])
        return reward

    def select_arm(self):
        """Return the machine index to take action on."""
        raise NotImplementedError
    
    def run_one_step(self, timestep, selected_arm, risk_score=-1):
        self.counts[selected_arm] += 1
        reward = self.update_estimates(selected_arm)
        updated_reward = self.update_reward(reward)
        updated_regret = self.update_regret(selected_arm)
        metric_dict = {
            'timestep': timestep,
            'risk_score': risk_score,
            'selected_arm': selected_arm,
            'reward': updated_reward,
            'regret': updated_regret
        }
        return metric_dict
        
    def run(self, num_steps, risk_sampler):
        metric_dicts = []
        timestep = 0

        # Try sampling each arm once to start off with
        for arm_idx in self.bandit.arm_indices:
            metric_dict = self.run_one_step(timestep, arm_idx)
            metric_dicts.append(metric_dict)
            timestep += 1

        for _ in range(num_steps):
            sample_risk_score = risk_sampler.sample_one()
            selected_arm = self.select_arm(sample_risk_score, timestep)
            metric_dict = self.run_one_step(timestep, selected_arm, sample_risk_score)
            metric_dicts.append(metric_dict)
            timestep += 1

        final_uncertainties = [self.get_uncertainty(arm, timestep) for arm in self.bandit.arm_indices]
        arms_dict = {'arm_idx': self.bandit.arm_indices,
                     'uncertainty_estimate': final_uncertainties,
                     'mean_estimate': self.estimates,
                     'count': self.counts}
        return pd.DataFrame(metric_dicts), pd.DataFrame(arms_dict)

class EpsilonGreedy(Solver):

    def __init__(self, bandit, epsilon, availability_type=None, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super().__init__(bandit, availability_type, init_proba)

        assert 0. <= epsilon <= 1.0
        self.epsilon = epsilon

    def select_arm(self, sample_risk_score, timestep):
        uncertainties = [self.get_uncertainty(arm, timestep) for arm in self.bandit.arm_indices]
        available_arms = self.availability_function(sample_risk_score, uncertainties)

        if np.random.random() < self.epsilon:
            # Let's do random exploration!
            selected_arm = np.random.choice(available_arms)
        else:
            # Pick the best one.
            selected_arm = max(available_arms, key=lambda x: self.estimates[x])

        return selected_arm


class UCB(Solver):
    '''
    UCB1
    '''
    def __init__(self, bandit, availability_type=None, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        """
        super().__init__(bandit, availability_type, init_proba)

    def select_arm(self, sample_risk_score, timestep):
        epsilon = 0.01

        # Pick the best one with consideration of upper confidence bounds.
        uncertainties = [self.get_uncertainty(arm, timestep) for arm in self.bandit.arm_indices]
        available_arms = self.availability_function(sample_risk_score, uncertainties)

        bounds = {arm: self.get_ucb(arm, timestep) for arm in available_arms}

        selected_arm = max(bounds, key=bounds.get)
        max_arms = [selected_arm]
        for arm, bound in bounds.items():
            if arm != selected_arm:
                if bound + epsilon > bounds[selected_arm]:
                    max_arms.append(arm)

        selected_arm = np.random.choice(max_arms)
        return selected_arm

    def get_ucb(self, arm, timestep):
        ucb = self.estimates[arm] + self.get_uncertainty(arm, timestep)
        return ucb


class LinUCB:
    def __init__(self, bandit, alpha=2.0, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        self.bandit = bandit
        self.alpha = alpha
        assert self.bandit.context is not None
        self.context_dim = self.bandit.context.shape[-1]

    def get_best_reward(self, X_timepoint):
        rewards = [self.bandit.select_arm(arm, X_timepoint[arm]) for arm in range(self.bandit.num_arms)]
        best_reward = np.max(rewards)
        return best_reward
    
    def get_random_reward(self, X_timepoint):
        random_arm = np.random.choice(self.bandit.num_arms)
        random_reward = self.bandit.select_arm(random_arm, X_timepoint[random_arm])
        return random_reward

    def run(self, num_timepoints):
        contexts = self.bandit.context
        
        oracle = np.empty(num_timepoints)
        rewards = np.empty(num_timepoints)
        random_rewards = np.empty(num_timepoints)
        selected_arms = np.empty(num_timepoints)
        theta = np.empty(shape=(num_timepoints, self.bandit.num_arms, self.context_dim))
        # expected reward
        predictions = np.empty(shape=(num_timepoints, self.bandit.num_arms))

        A_matrix = [np.identity(self.context_dim) for arm in range(self.bandit.num_arms)]
        b_vector = [np.zeros(self.context_dim) for arm in range(self.bandit.num_arms)] 

        for timestep in range(num_timepoints):
            # For each arm, calculate theta (regression coefficient) and ucb (confidence bound)
            for arm_idx in range(self.bandit.num_arms):
                A = A_matrix[arm_idx]
                b = b_vector[arm_idx]
                A_inv = np.linalg.inv(A)
                theta[timestep, arm_idx] = np.dot(A_inv, b)

                # Context for timestep, arm_idx
                X_ta = contexts[timestep, arm_idx, :]

                predictions[timestep, arm_idx] = \
                    np.dot(theta[timestep, arm_idx].T, X_ta) + (self.alpha * np.sqrt(np.dot(X_ta.T, np.dot(A_inv, X_ta))))
                
                

            # Select arm with highest confidence bound
            selected_arm = np.argmax(predictions[timestep])
            X_selected_arm = contexts[timestep, selected_arm, :]

            rewards[timestep] = self.bandit.select_arm(selected_arm, X_selected_arm)
            selected_arms[timestep] = selected_arm 

            # Get oracle
            oracle[timestep] = self.get_best_reward(contexts[timestep, :, :])
            random_rewards[timestep] = self.get_random_reward(contexts[timestep, :, :])

            # Update A_matrix and b_matrix
            A_matrix[selected_arm] += np.outer(X_selected_arm, X_selected_arm.T)
            b_vector[selected_arm] += rewards[timestep] * X_selected_arm


        return theta, predictions, selected_arms, rewards, oracle, random_rewards
