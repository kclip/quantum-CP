import torch
import numpy as np
import random
import warnings

#### basic utils
def reset_random_seed(random_seed):
    if_fix_random_seed = True
    if if_fix_random_seed:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
    else:
        pass

def quantile_plus(NC_vec, tau):
    assert len(NC_vec.shape) == 1 # (N+1)
    N = NC_vec.shape[0] # this is actually N+1 in our notations
    sorted_vec_x, _ = torch.sort(NC_vec, dim=0)
    ind_quantile = np.ceil((tau)*(N)) - 1 ### python index starts from 0
    return sorted_vec_x[int(ind_quantile)]


def rounding_x(x):
    return np.round(x.numpy(), decimals=int(3)) # this is just to make sure nothing bad happend for evaluation


