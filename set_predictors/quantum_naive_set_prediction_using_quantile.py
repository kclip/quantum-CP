import torch
import numpy as np
from funcs.basic_utils import quantile_plus
import scipy
from scipy.stats import norm, t
from funcs.basic_utils import rounding_x
from set_predictors.training_PQC_toy_without_input import PQC_Trainer_without_input
from set_predictors.training_PQC_toy_with_input import PQC_Trainer_with_input
import copy
import random
#from qiskit_ibm_runtime import Sampler

class Naive_Predictor: # naive quantile
    def __init__(self, alpha, pqc_model, M, bin_num_for_naive_histo, exp_mode): # M: number of shots
        self.alpha = alpha
        self.pqc_model = pqc_model
        if pqc_model is None:
            pass
        else:
            self.pqc_model_init_info = (pqc_model.num_layers, pqc_model.num_qubits, pqc_model.observation_min_max, pqc_model.quantum_para_list)
        self.M = M
        self.curr_fixed_measurement = None
        self.bin_num_for_naive_histo = bin_num_for_naive_histo
        self.exp_mode = exp_mode
        self.already_trained = False
        self.curr_fixed_measurement_dict = None
    def fix_measurement_with_given_list(self, curr_fixed_measurement):
        self.curr_fixed_measurement = curr_fixed_measurement
    def fix_measurement_with_given_dict_per_input(self, curr_fixed_measurement_dict):
        # curr_fixed_measurement_dict: 'val': measurements list, 'te': meausrements list
        self.curr_fixed_measurement_dict = curr_fixed_measurement_dict # this changes every indep. exp!

    def train(self, exp_mode, iter_cont_pqc, M_tr, training_mode, tr_dataset, num_tr_iter, lr, angle_encoding_mode_for_with_input=None, bimodal_mode_unsup=None): 
        if M_tr is None:
            M_tr = self.M
        else:
            pass
        if exp_mode == 'toy_without_input':
            trainer = PQC_Trainer_without_input(iter_cont_pqc, self.pqc_model, M_tr, training_mode, tr_dataset, num_tr_iter, lr, bimodal_mode_unsup)
        else:
            trainer = PQC_Trainer_with_input(iter_cont_pqc, self.pqc_model, M_tr, training_mode, tr_dataset, num_tr_iter, lr, angle_encoding_mode_for_with_input)
        trainer.forward()

    def set_prediction(self, val_dataset, test_inputs):
        dict_set_pred_test_inputs = self.set_prediction_wrapper(test_inputs, None)
        return dict_set_pred_test_inputs

    def set_prediction_wrapper(self, test_inputs, sampler):
        dict_set_pred_test_inputs = {}
        ind_x_te = 0
        for x_te in test_inputs:
            if self.curr_fixed_measurement_dict is None:  
                pass
            else:
                self.curr_fixed_measurement = self.curr_fixed_measurement_dict['te'][ind_x_te][:self.M]
            histogram_augmented_data_dict = self.make_histo_for_fixed_input(x_te, sampler)
            curr_set_pred_for_x_te, curr_ineff = compute_set_prediction(histogram_augmented_data_dict, self.alpha)
            dict_set_pred_test_inputs['predicted_set_for_input_'+ str(rounding_x(x_te))+'_ind_'+str(ind_x_te)] = curr_set_pred_for_x_te
            dict_set_pred_test_inputs['inefficiency_for_input_'+ str(rounding_x(x_te))+'_ind_'+str(ind_x_te)] = curr_ineff
            ind_x_te += 1
        return dict_set_pred_test_inputs

    def make_histo_for_fixed_input(self, x, sampler):
        if self.curr_fixed_measurement is None:
            measurement_list = self.pqc_model.M_shots_measurement_single(x, self.M, sampler)
        else:
            measurement_list = self.curr_fixed_measurement
        Y_augmented = torch.tensor(measurement_list).unsqueeze(dim=1)
        if self.bin_num_for_naive_histo is None:
            if self.pqc_model is None:
                self.bin_num = 32
            else:
                self.bin_num = 2**self.pqc_model.num_qubits 
        else:
            self.bin_num = self.bin_num_for_naive_histo
        num_augmented = Y_augmented.shape[0]
        sorted_Y_aug, _ = torch.sort(Y_augmented, dim=0) # small to large
        measure, intervals = torch.histogram(Y_augmented.squeeze(), bins= self.bin_num, density=True)
        interval_width = intervals[1]-intervals[0]
        measure *= interval_width # make it as prob. measure
        histogram_augmented_data_dict = {}
        for ind_bin in range(self.bin_num):
            histogram_augmented_data_dict[( intervals[ind_bin], intervals[ind_bin+1] )] = measure[ind_bin]
        return histogram_augmented_data_dict

def compute_set_prediction(histogram_augmented_data_dict, alpha):
    #histogram_augmented_data_dict = {'bin_1': 0.4, 'bin_2': 0.1, 'bin_3': 0.00,  'bin_4': 0.05, 'bin_5':0.05, 'bin_6': 0.05, 'bin_7':0.25, 'bin_9':0.1, 'bin_10':0.00}
    sorted_dict = {k: v for k, v in sorted(histogram_augmented_data_dict.items(), key=lambda item: item[1], reverse=True)}
    accum_empirical_histogram = 0
    ind_interval = 0
    union_of_intervals = []
    ineff = 0
    while accum_empirical_histogram < 1-alpha:
        curr_interval = list(sorted_dict.keys())[ind_interval]
        union_of_intervals.append(curr_interval) # curr_interval: [-1, -0.5]...
        ineff += (curr_interval[1]-curr_interval[0])
        accum_empirical_histogram += sorted_dict[curr_interval]
        if accum_empirical_histogram >= 1-alpha: # then it means that current interval (empirical) prob. measure crossed the 1-alpha, so we are putting the intervals with the same measure
            # check next interval (since it is already sorted)
            if ind_interval == len(sorted_dict.keys())-1: # reached at the dictionary
                assert accum_empirical_histogram == 1.0
                #return union_of_intervals, accum_empirical_histogram
                pass
            else:
                same_measure_ind = ind_interval + 1
                next_interval = list(sorted_dict.keys())[same_measure_ind]
                candidate_measure = sorted_dict[next_interval]
                while candidate_measure == sorted_dict[curr_interval]:
                    union_of_intervals.append(next_interval)
                    ineff += (next_interval[1]-next_interval[0])
                    accum_empirical_histogram += candidate_measure
                    # check next interval (since it is already sorted)
                    if same_measure_ind == len(sorted_dict.keys())-1: # reached at the dictionary
                        assert torch.norm(accum_empirical_histogram -1.0) < 0.0001 # this means that we are having entire set at predicted set
                        break
                    else:
                        same_measure_ind += 1
                        next_interval = list(sorted_dict.keys())[same_measure_ind]
                        candidate_measure = sorted_dict[next_interval]
        else:
            pass
        ind_interval += 1
    return union_of_intervals, ineff