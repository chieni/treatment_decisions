import numpy as np
from enum import Enum


class NormalDist():
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self.std = np.sqrt(self.variance)
    
    @classmethod
    def generate_random_dist(cls):
        mean = np.random.uniform(-1, 1)
        var = np.random.uniform(0.5, 1.5)
        return cls(mean, var)

class AvailabilityType(Enum):
    first_fraction = 'first_fraction'
    mean_estimates = 'mean_estimates'
    uncertainty_estimates = 'uncertainty_estimates'