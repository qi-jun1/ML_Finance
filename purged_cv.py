from sklearn.model_selection import KFold
import numpy as np
from itertools import product

def purged_kfold_indices(n_samples, n_splits=5, purge_size=0):
    """
    Generate train-test indices for Purged k-Fold Cross Validation.

    Parameters:
    - n_samples (int): Total number of samples.
    - n_splits (int): Number of folds.
    - purge_size (int): Number of samples to exclude before and after the test set.

    Returns:
    - List of (train_indices, test_indices) tuples.
    """
    kf = KFold(n_splits=n_splits, shuffle=False)
    indices = np.arange(n_samples)

    folds = []
    for train_idx, test_idx in kf.split(indices):
        test_start, test_end = test_idx[0], test_idx[-1]

        # Define the purge window
        purge_start = max(0, test_start - purge_size)
        purge_end = min(n_samples, test_end + purge_size + 1)

        # Remove purged indices from training set
        train_idx = np.setdiff1d(train_idx, np.arange(purge_start, purge_end))

        folds.append((train_idx, test_idx))

    return folds


def param_grid_dicts(param_dict):
    keys = list(param_dict.keys())
    values_product = product(*[param_dict[k] for k in keys])
    for combo in values_product:
        yield dict(zip(keys, combo))
