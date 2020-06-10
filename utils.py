import math
import numpy as np
import os
import pickle
import random

from loss_functions import safe_sparse_dot, safe_sparse_norm


def relative_round(x):
    """
    A util that rounds the input to the most significant digits.
    Useful for storing the results as rounding float
    numbers may cause file name ambiguity.
    """
    mantissa, exponent = math.frexp(x)
    return round(mantissa, 3) * 2**exponent

    
def get_trace(path, loss):
    if not os.path.isfile(path):
        return None
    f = open(path, 'rb')
    trace = pickle.load(f)
    trace.loss = loss
    f.close()
    return trace


def set_seed(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    

def variance_at_opt(x_opt, loss, batch_size=1, perms=None, n_perms=1, L_batch=None, lr=None):
    if L_batch is None:
        # for simplicity, we use max smoothness, but one might want to use batch smoothness instead
        L_batch = loss.max_smoothness()
    if lr is None:
        lr = 1 / L_batch
    if perms is None:
        perms = [np.random.permutation(loss.n) for _ in range(n_perms)]
    else:
        n_perms = len(perms)
    variance_sgd = 0
    variances = None
    for permutation in perms:
        grad_sum = 0
        start_idx = range(0, loss.n, batch_size)
        n_grads = len(start_idx)
        if variances is None:
            variances = np.zeros(n_grads)
        for e, i in enumerate(start_idx):
            idx = permutation[np.arange(i, min(loss.n, i + batch_size))]
            stoch_grad = loss.stochastic_gradient(x_opt, idx=idx)
            variance_sgd += safe_sparse_norm(stoch_grad)**2 / n_grads / n_perms
            
            x = x_opt - lr * grad_sum
            loss_x = loss.partial_value(x, idx)
            loss_x_opt = loss.partial_value(x_opt, idx)
            linear_dif = safe_sparse_dot(stoch_grad, x - x_opt)
            bregman_div = loss_x - loss_x_opt - linear_dif
            variances[e] += bregman_div / n_perms / lr
            grad_sum += stoch_grad
    variance_rr = np.max(variances)
    variance_rr_upper = variance_sgd * n_grads * lr * L_batch / 4
    variance_rr_lower = variance_sgd * n_grads * lr * loss.l2 / 8
    return variance_sgd, variance_rr, variance_rr_upper, variance_rr_lower
