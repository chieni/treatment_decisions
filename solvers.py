import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import NormalDist, AvailabilityType


class Solver:
    def __init__(self, bandit, availability_type, init_proba):
        """
        bandit (Bandit): the target bandit to solve.
        """
        self.bandit = bandit
        self.availability_type = availability_type

        self.counts = [0] * self.bandit.num_arms
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.

        # self.estimates is the empirical mean of the observed reward samples
        self.estimates = [init_proba] * self.bandit.num_arms # Optimistic initialization
        self.timestep = 0

    def update_regret(self, selected_arm):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_reward_prob - self.bandit.reward_probs[selected_arm]
        self.regrets.append(self.regret)

    def availability_function(self, sample_risk_score, uncertainties):
        all_arms_indices = self.bandit.arm_indices
        num_arms = self.bandit.num_arms

        # Select first fraction of arms based on risk score
        if self.availability_type == AvailabilityType.first_fraction:
            end_idx = max(1, int(sample_risk_score * num_arms))
            return all_arms_indices[:end_idx]
        
        if self.availability_type == AvailabilityType.mean_estimates:
            # Note: if everything is equal, do not sort the array, otherwise the array just flips
            if len(set(self.estimates)) == 1:
                sorted_estimates = list(all_arms_indices)
            else:
                sorted_estimates = np.argsort(self.estimates)[::-1]
            end_idx = max(1, int(sample_risk_score * num_arms))
            return sorted_estimates[:end_idx]

        if self.availability_type == AvailabilityType.uncertainty_estimates:
            if len(set(uncertainties)) == 1:
                sorted_uncertainties = list(all_arms_indices)
            else:
                sorted_uncertainties = np.argsort(uncertainties)
            end_idx = max(1, int(sample_risk_score * num_arms))
            return sorted_uncertainties[:end_idx]
            
        return all_arms_indices
    
    def get_uncertainty(self, arm):
        return np.sqrt(2. * np.log(self.timestep) / (self.counts[arm] + 1)) 

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps, risk_sampler):
        metric_dicts = []
        for _ in range(num_steps):
            sample_risk_score = risk_sampler.sample_one()
            selected_arm = self.run_one_step(sample_risk_score)

            self.counts[selected_arm] += 1
            self.actions.append(selected_arm)
            self.update_regret(selected_arm)

            metric_dicts.append({
                'risk_score': sample_risk_score,
                'selected_arm': selected_arm
            })
        return pd.DataFrame(metric_dicts)

class EpsilonGreedy(Solver):

    def __init__(self, bandit, epsilon, availability_type=None, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super().__init__(bandit, availability_type, init_proba)

        assert 0. <= epsilon <= 1.0
        self.epsilon = epsilon

    def run_one_step(self, sample_risk_score):
        self.timestep += 1
        uncertainties = [self.get_uncertainty(arm) for arm in self.bandit.arm_indices]
        available_arms = self.availability_function(sample_risk_score, uncertainties)

        if np.random.random() < self.epsilon:
            # Let's do random exploration!
            selected_arm = np.random.choice(available_arms)
            #selected_arm = np.random.randint(0, self.bandit.num_arms)
        else:
            # Pick the best one.
            selected_arm = max(available_arms, key=lambda x: self.estimates[x])
            #selected_arm = max(range(self.bandit.num_arms), key=lambda x: self.estimates[x])

        reward = self.bandit.select_arm(selected_arm)
        self.estimates[selected_arm] += 1. / (self.counts[selected_arm] + 1) * (reward - self.estimates[selected_arm])

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

    def run_one_step(self, sample_risk_score):
        self.timestep += 1

        # Pick the best one with consideration of upper confidence bounds.
        # bounds = [self.get_ucb(arm) for arm in range(self.bandit.num_arms)]
        # selected_arm = np.argmax(bounds)
        uncertainties = [self.get_uncertainty(arm) for arm in self.bandit.arm_indices]
        available_arms = self.availability_function(sample_risk_score, uncertainties)

        bounds = {arm: self.get_ucb(arm) for arm in available_arms}
        selected_arm = max(bounds, key=bounds.get)
        
        reward = self.bandit.select_arm(selected_arm)
        self.estimates[selected_arm] += 1. / (self.counts[selected_arm] + 1) * (reward - self.estimates[selected_arm])

        #self.estimates[selected_arm] = (reward + (self.estimates[selected_arm] * self.counts[selected_arm])) / (self.counts[selected_arm] + 1)
        return selected_arm

    def get_ucb(self, arm):
        # if self.counts[arm] == 0:
        #     return math.inf
        ucb = self.estimates[arm] + self.get_uncertainty(arm)
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
