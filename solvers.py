import numpy as np
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, bandit, init_proba):
        """
        bandit (Bandit): the target bandit to solve.
        """
        self.bandit = bandit

        self.counts = [0] * self.bandit.num_arms
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.
        self.estimates = [init_proba] * self.bandit.num_arms # Optimistic initialization

    def update_regret(self, selected_arm):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_reward_prob - self.bandit.reward_probs[selected_arm]
        self.regrets.append(self.regret)

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            selected_arm = self.run_one_step()

            self.counts[selected_arm] += 1
            self.actions.append(selected_arm)
            self.update_regret(selected_arm)

class EpsilonGreedy(Solver):

    def __init__(self, bandit, epsilon, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super().__init__(bandit, init_proba)

        assert 0. <= epsilon <= 1.0
        self.epsilon = epsilon

        self.estimates = [init_proba] * self.bandit.num_arms # Optimistic initialization

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            # Let's do random exploration!
            selected_arm = np.random.randint(0, self.bandit.num_arms)
        else:
            # Pick the best one.
            selected_arm = max(range(self.bandit.num_arms), key=lambda x: self.estimates[x])

        r = self.bandit.select_arm(selected_arm)
        self.estimates[selected_arm] += 1. / (self.counts[selected_arm] + 1) * (r - self.estimates[selected_arm])

        return selected_arm

class UCB(Solver):
    def __init__(self, bandit, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super().__init__(bandit, init_proba)
        self.estimates = [init_proba] * self.bandit.num_arms # Optimistic initialization

        self.threshold = 0

    def run_one_step(self):
        self.threshold += 1

        # Pick the best one with consideration of upper confidence bounds.
        bounds = [self.get_ucb(arm) for arm in range(self.bandit.num_arms)]
        selected_arm = np.argmax(bounds)
        reward = self.bandit.select_arm(selected_arm)

        #self.estimates[selected_arm] += 1. / (self.counts[selected_arm] + 1) * (reward - self.estimates[selected_arm])

        self.estimates[selected_arm] = (reward + (self.estimates[selected_arm] * self.counts[selected_arm])) / (self.counts[selected_arm] + 1)
        return selected_arm
    
    def get_ucb(self, arm):
        ucb = self.estimates[arm] + np.sqrt(2. * np.log(self.threshold) / (self.counts[arm] + 1))
        return ucb


class LinUCB(Solver):
    def __init__(self, bandit, alpha=2.0, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super().__init__(bandit, init_proba)
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
