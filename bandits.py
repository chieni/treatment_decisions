'''
Used as reference:
https://github.com/lilianweng/multi-armed-bandit
https://www.kaggle.com/phamvanvung/cb-linucb/comments
http://snap.stanford.edu/class/cs246-2015/slides/18-bandits.pdf
'''
import numpy as np


class BernoulliBandit():
    def __init__(self, num_arms, reward_probs=None):
        self.num_arms = num_arms

        if reward_probs is None:
            self.reward_probs = [np.random.random() for _ in range(self.num_arms)]
        else:
            self.reward_probs = reward_probs
        self.best_reward_prob = max(self.reward_probs)
        
    def select_arm(self, arm_idx):
        if np.random.random() < self.reward_probs[arm_idx]:
            return 1
        return 0

class ContextualBandit():
    def __init__(self, num_arms, context, true_theta):
        self.num_arms = num_arms
        self.context = context
        self.true_theta = true_theta
        
    def select_arm(self, arm_idx, value, scale_noise=0.1):
        signal = self.true_theta[arm_idx].dot(value)
        noise = np.random.normal(scale=scale_noise)
        return (signal + noise)
