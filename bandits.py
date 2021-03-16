'''
Used as reference:
https://github.com/lilianweng/multi-armed-bandit
https://www.kaggle.com/phamvanvung/cb-linucb/comments
http://snap.stanford.edu/class/cs246-2015/slides/18-bandits.pdf
'''
import numpy as np
from utils import NormalDist



class BernoulliBandit():
    def __init__(self, num_arms, reward_probs=None):
        self.num_arms = num_arms

        if reward_probs is None:
            self.reward_probs = [np.random.random() for _ in range(self.num_arms)]
        else:
            self.reward_probs = reward_probs
        self.best_reward_prob = max(self.reward_probs)
        self.reward_variances = None
        
    def select_arm(self, arm_idx):
        '''
        Binary reward based on self.reward_probs set during initialization.
        '''
        if np.random.random() < self.reward_probs[arm_idx]:
            return 1
        return 0

class GaussianBandit():
    def __init__(self, num_arms, reward_dists=None):
        # Means and variances represented dist of an arm?
        self.num_arms = num_arms
        self.arm_indices = [idx for idx in range(self.num_arms)]

        if reward_dists is None:
            self.reward_dists = self._generate_random_rewards(num_arms)
        else:
            self.reward_dists = reward_dists
        self.reward_probs = [dist.mean for dist in self.reward_dists]
        self.best_reward_prob = max(self.reward_probs)
        self.reward_variances = [dist.variance for dist in self.reward_dists]
    
    def _generate_random_rewards(self, num_arms):
        return [NormalDist.generate_random_dist() for arm in num_arms]

    def select_arm(self, arm_idx):
        '''
        Sample from normal distribution of arm.
        '''
        return np.random.normal(loc=self.reward_dists[arm_idx].mean, scale=self.reward_dists[arm_idx].std)

class ContextualBandit():
    def __init__(self, num_arms, context, true_theta):
        self.num_arms = num_arms
        self.context = context
        self.true_theta = true_theta
        
    def select_arm(self, arm_idx, value, scale_noise=0.1):
        '''
        value: context
        '''
        signal = self.true_theta[arm_idx].dot(value)
        noise = np.random.normal(scale=scale_noise)
        return (signal + noise)
