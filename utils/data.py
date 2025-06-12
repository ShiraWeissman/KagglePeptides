import json
import os
import pandas as pd
import numpy as np
import pickle

root_path = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))


dataset_path = os.path.join(root_path, 'data\dataset')

def load_data(data_path):
    if data_path.split('.')[1] == 'csv':
        with open(data_path, 'r') as f:
            data = pd.read_csv(f)
    return data

def save_data(data, data_path, file_type):
    if file_type == 'csv':
        with open(data_path, 'w') as f:
            data.to_csv(f, index=False)
    elif file_type == 'json':
        with open(data_path, 'w') as f:
            json.dump(data, f)

if __name__ == '__main__':
    pass