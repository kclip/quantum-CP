import torch
import os
import matplotlib.pyplot as plt
import warnings
import numpy as np
from funcs.basic_utils import rounding_x
from scipy.stats import norm

def cov_ineff_compute(dict_set_pred_test_inputs, test_dataset):
    total_coverage = 0
    total_ineff = 0
    set_prediction_dict = {}
    X_te = test_dataset[0] # (num_te, dim_x)
    Y_te = test_dataset[1] # (num_te, 1)
    num_te = X_te.shape[0]
    for ind_te in range(num_te):
        x_te = X_te[ind_te]
        y_te = Y_te[ind_te]
        if not type(y_te) == torch.Tensor:   # y_te is actually the class, since we are taking expectation over the targer distribution to compute coverage for unsupervised learning case (data generator)
            assert x_te == 0.0 # only for unsupervised learning case
            curr_set_pred_for_x_te = dict_set_pred_test_inputs['predicted_set_for_input_'+ str(rounding_x(x_te))+'_ind_'+str(ind_te)]
            curr_ineff = dict_set_pred_test_inputs['inefficiency_for_input_'+str(rounding_x(x_te))+'_ind_'+str(ind_te)]
            total_ineff += curr_ineff
            curr_coverage = 0
            for curr_interval in curr_set_pred_for_x_te:
                curr_coverage += y_te.compute_coverage_for_single_interval(curr_interval)
            total_coverage += curr_coverage
        else:
            curr_set_pred_for_x_te = dict_set_pred_test_inputs['predicted_set_for_input_'+ str(rounding_x(x_te))+'_ind_'+str(ind_te)]
            curr_ineff = dict_set_pred_test_inputs['inefficiency_for_input_'+str(rounding_x(x_te))+'_ind_'+str(ind_te)]
            total_ineff += curr_ineff
            for grid_interval_in_set_pred in curr_set_pred_for_x_te:
                if (y_te >= grid_interval_in_set_pred[0]) and (y_te <= grid_interval_in_set_pred[1]):
                    total_coverage += 1
                    break
                else:
                    pass
    total_coverage /= num_te
    if total_ineff == 0:
        total_ineff = torch.tensor(0.0)
    total_ineff /= num_te
    return total_coverage, total_ineff
    