# src1/cross_entropy_validator.py

import numpy as np
import pandas as pd

class CrossEntropyAnalyzer:
    def __init__(self, steps_data):
        self.steps_data = steps_data
        self.cross_entropy_series = None

    def compute_cross_entropy(self):
        def entropy(p, q):
            p = np.clip(p, 1e-10, 1.0)
            q = np.clip(q, 1e-10, 1.0)
            return -np.sum(p * np.log(q))

        inner_bins = [f'ep_inner_bin_{i}' for i in range(11)]
        outer_bins = [f'ep_outer_bin_{i}' for i in range(11)]
        entropies = []

        for idx, row in self.steps_data.iterrows():
            p = row[inner_bins].values
            q = row[outer_bins].values
            if np.any(p <= 0) or np.any(q <= 0):
                entropies.append(np.nan)
                continue
            p /= p.sum()
            q /= q.sum()
            entropies.append(entropy(p, q))

        self.steps_data['cross_entropy_inner_outer'] = entropies
        self.cross_entropy_series = self.steps_data['cross_entropy_inner_outer']

    def get_high_entropy_events(self, threshold=2.0):
        return self.cross_entropy_series[self.cross_entropy_series > threshold].index
