import json
import os
import argparse
import yaml
import pandas as pd
import numpy as np
import random
import torch
from utils.data import *
root_path = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))

saved_models_path = os.path.join(root_path, r'pretrained_models')

best_scheme_path = os.path.join(root_path, r'results\best_schemes')

predictions_path = os.path.join(root_path, r'results\predictions')

def verify_folder(file_path):
    if '.' in file_path:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
    else:
        if not os.path.exists(file_path):
            os.makedirs(file_path)


def train_test_split(X, y, train_test_ratio=0.2, random_state=0):
    random.seed(random_state)
    idx_list = list(range(y.shape[0]))
    random.shuffle(idx_list)
    test_idx = idx_list[:int(train_test_ratio*len(idx_list))]
    train_idx = idx_list[int(train_test_ratio*len(idx_list)):]
    return X[train_idx, :], y[train_idx, :], X[test_idx, :], y[test_idx, :], train_idx, test_idx


def save_predictions(predictions, predictions_path):
    predictions = verify_json_form(predictions)
    verify_folder(predictions_path)
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f)

def verify_json_form(param):
    if type(param) in [int, float, str]:
        return param
    elif type(param) == np.ndarray:
        param = param.tolist()
        return param
    elif torch.is_tensor(param):
        param = param.detach().numpy().tolist()
        return param
    elif type(param) == list:
        for i, p in enumerate(param.copy()):
            param[i] = verify_json_form(p)
        return param
    elif type(param) == dict:
        param_copy = param.copy()
        for k in param_copy:
            param[k] = verify_json_form(param_copy[k])
        return param

    else:
        return str(param)

def are_dicts_identical(dict1, dict2):
    """
    Recursively checks if two nested dictionaries are identical.
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict1 == dict2

    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        if not are_dicts_identical(dict1[key], dict2[key]):
            return False

    return True

if __name__ == '__main__':
    pass
    # res = verify_json_form([1, {'a': np.array([1,2,3])}, 3, torch.stack([torch.tensor([2,3,4]), torch.tensor([2,3,4])])])
    # print(json.dumps(res))
    # # 1, {'a': 2}, 3,
    # print()