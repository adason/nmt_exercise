""" Common Mathematical Transformation Functions.
"""
import numpy as np


def binarize(y, n_cols):
    """ Convert multi-class y labels into one-hot encoded vectors.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        Indices of nonzero elements
    n_cols : int
        Dimension of the banalized vector

    Returns
    -------
    numpy array of shape (n_samples, n_features)
        Array of one-hot encoded vectors.
    """
    y = np.array(y)
    n_samples = y.shape[0]
    bin_matrix = np.zeros((n_samples, n_cols))
    for i, idx in enumerate(y):
        bin_matrix[i, idx] = 1

    return bin_matrix


def softmax(s, axis=None):
    """ Compute the softmax transformation along a given axis.

    Note
    ----
        Formula: p_i = exp(s_i - max(s)) / \sum_i exp(s_i - max(s))

    Parameters
    ----------
    s : numpy array of any shape
        Input
    axis : int, optional
        The axis to perform softmax transformation, defaults to the last dimension.

    Returns
    -------
    numpy array of the same shape of input array
        Softmax transformed numpy array
    """
    s = np.atleast_2d(s)

    if axis is None:
        axis = s.ndim - 1
    s = s - np.expand_dims(np.max(s, axis), axis)
    s = np.exp(s)
    s_sum = np.expand_dims(np.sum(s, axis), axis)
    p = s / s_sum

    return p.squeeze()
