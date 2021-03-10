import numpy as np


np.random.seed(0)
class RiskSampler():
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high
    
    def sample_one(self):
        return np.random.uniform(self.low, self.high)
