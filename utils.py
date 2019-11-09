import json
import numpy as np
import pandas as pd

class Params():
    """Parameters object taken from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
    
    Parameters
    ----------
    json_path : string

    Returns
    ----------
    Parameters object
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def one_hot_encode(y):
    """ One hot encode y.  We use this to get from 1 dim ys to predict proba's.
    This is taken from this s.o. post: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array

    Parameters
    ----------
    y : np.ndarray

    Returns
    ----------
    A np.ndarray of the one hot encoded data.
    """
    y_hat_one_hot = np.zeros((len(y), 2))
    y_hat_one_hot[np.arange(len(y)), y] = 1
    return y_hat_one_hot
