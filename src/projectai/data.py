import os
import pandas as pd


def _repo_root(from_path=None):
    if from_path is None:
        from_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.abspath(from_path)


def find_data_paths(repo_root=None):
    """Return (train_path, test_path) searching common locations.

    Order: `data/raw/train.csv`, `house-prices-advanced-regression-techniques/train.csv`.
    """
    repo_root = _repo_root(repo_root)
    candidates = [
        os.path.join(repo_root, 'data', 'raw', 'train.csv'),
        os.path.join(repo_root, 'house-prices-advanced-regression-techniques', 'train.csv'),
    ]
    for train_p in candidates:
        if os.path.exists(train_p):
            test_p = os.path.join(os.path.dirname(train_p), 'test.csv')
            return train_p, test_p
    raise FileNotFoundError('train.csv not found in data/raw or house-prices-advanced-regression-techniques')


def load_train_test(repo_root=None):
    train_p, test_p = find_data_paths(repo_root)
    train = pd.read_csv(train_p)
    test = pd.read_csv(test_p)
    return train, test
