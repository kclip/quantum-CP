import torch
import numpy as np
from funcs.basic_utils import quantile_plus, rounding_x
import copy
import os
import platform
from set_predictors.training_PQC_toy_without_input import PQC_Trainer_without_input
from set_predictors.training_PQC_toy_with_input import PQC_Trainer_with_input

class QCP:
    def __init__(self, alpha, pqc_model, M, NC_mode, exp_mode=None): 
        self.alpha = alpha
        self.pqc_model = pqc_model
        if pqc_model is None:
            pass
        else:
            self.pqc_model_init_info = (pqc_model.num_layers, pqc_model.num_qubits, pqc_model.observation_min_max, pqc_model.quantum_para_list)
        self.M = M
        self.NC_mode = NC_mode
        self.curr_fixed_measurement = None
        self.already_trained = False
        self.curr_fixed_measurement_dict = None

    def fix_measurement_with_given_list(self, curr_fixed_measurement):
        self.curr_fixed_measurement = curr_fixed_measurement

    def fix_measurement_with_given_dict_per_input(self, curr_fixed_measurement_dict):
        # curr_fixed_measurement_dict: 'val': measurements list, 'te': meausrements list
        self.curr_fixed_measurement_dict = curr_fixed_measurement_dict # this changes every indep. exp!

    def train(self, exp_mode, M_tr, training_mode, tr_dataset, num_tr_iter, lr, angle_encoding_mode_for_with_input=None, bimodal_mode_unsup=None): 
        if M_tr is None:
            M_tr = self.M
        else:
            pass
        if exp_mode == 'toy_without_input':
            trainer = PQC_Trainer_without_input(self.pqc_model, M_tr, training_mode, tr_dataset, num_tr_iter, lr, bimodal_mode_unsup)
        else:
            trainer = PQC_Trainer_with_input(self.pqc_model, M_tr, training_mode, tr_dataset, num_tr_iter, lr, angle_encoding_mode_for_with_input)
        trainer.forward()
        
    def set_prediction(self, val_dataset, test_inputs):
        dict_set_pred_test_inputs = self.set_prediction_wrapper(val_dataset, test_inputs, None)
        return dict_set_pred_test_inputs

    def set_prediction_wrapper(self, val_dataset, test_inputs, sampler):
        # compute (1-alpha)-quantile
        NC_val_list = []
        if val_dataset[0] is None:
            num_val = 0
        else:
            num_val = val_dataset[0].shape[0]
        for ind_val in range(num_val):
            val_example = ( val_dataset[0][ind_val], val_dataset[1][ind_val] )
            if self.curr_fixed_measurement_dict is None:  
                pass
            else: ## for the supervised learning case since meausrements depend on input now
                self.curr_fixed_measurement = self.curr_fixed_measurement_dict['val'][ind_val][:self.M]
            curr_NC_val, _ = self.compute_NC(self.NC_mode, val_example, self.pqc_model, self.M, sampler, self.curr_fixed_measurement)
            NC_val_list.append(curr_NC_val)
        if num_val == 0:
            NC_val_list.append(torch.tensor([9999999999])) # add infty to compute 1-alpha quantile
        else:
            NC_val_list.append(9999999999 * torch.ones(curr_NC_val.shape) ) # add infty to compute 1-alpha quantile
        NC_vec = torch.cat(NC_val_list, dim=0) # [(1),...,(1)] ->  (N+1)        
        NC_quantile = quantile_plus(NC_vec, 1-self.alpha)
        dict_set_pred_test_inputs = {}
        ind_x_te = 0
        for x_te in test_inputs:
            if self.curr_fixed_measurement_dict is None:  
                pass
            else: ## for with input case!
                self.curr_fixed_measurement = self.curr_fixed_measurement_dict['te'][ind_x_te][:self.M]
            curr_set_pred_for_x_te, curr_ineff = self.compute_set_prediction(x_te, NC_quantile, self.NC_mode, sampler)
            dict_set_pred_test_inputs['predicted_set_for_input_'+ str(rounding_x(x_te))+'_ind_'+str(ind_x_te)] = curr_set_pred_for_x_te
            dict_set_pred_test_inputs['inefficiency_for_input_'+ str(rounding_x(x_te))+'_ind_'+str(ind_x_te)] = curr_ineff
            ind_x_te += 1
        return dict_set_pred_test_inputs

    def compute_set_prediction(self, x_te, NC_quantile, NC_mode, sampler):
        ineff = 0
        set_prediction_list = []
        dummy_y_value = 0 # we are not computing NC but just using the output from x_te
        _, output_from_x_te = self.compute_NC(self.NC_mode, (x_te, dummy_y_value), self.pqc_model, self.M, sampler, self.curr_fixed_measurement)
        if NC_mode == 'distance_aware':
            y_hat_te = output_from_x_te
            single_interval = (y_hat_te - NC_quantile, y_hat_te + NC_quantile)
            set_prediction_list.append(single_interval)
            ineff = single_interval[1] - single_interval[0]
            return set_prediction_list, ineff
        elif NC_mode == 'distribution_aware': # k(M)=1 (PCP case, see also eq 20 in the paper)
            measurement_list_te = output_from_x_te # [0.7, 0.3, ...]
            measurement_list_te.sort() # small to large
            union_of_intervals = []
            ind_measurement = 0
            ineff = 0
            for single_measurement in measurement_list_te:
                curr_interval = [single_measurement-NC_quantile, single_measurement+NC_quantile]
                if ind_measurement == 0:
                    expanding_curr_interval = curr_interval
                else:
                    if curr_interval[0] <= expanding_curr_interval[1]:
                        expanding_curr_interval[1] = curr_interval[1]
                    else:
                        union_of_intervals.append(expanding_curr_interval)
                        ineff += (expanding_curr_interval[1]-expanding_curr_interval[0])
                        expanding_curr_interval = curr_interval
                ind_measurement += 1
            union_of_intervals.append(expanding_curr_interval) # last one 
            ineff += (expanding_curr_interval[1]-expanding_curr_interval[0])
            return union_of_intervals, ineff
        elif 'distribution_aware' in NC_mode: # k(M)= some integer that can be larger than 1 (PCP case, see also eq 21 in the paper)
            ineff = 0
            union_of_intervals = []
            K_nn = int(NC_mode[21:])
            # first get all the possible partitions
            measurement_list_te = output_from_x_te # [0.7, 0.3, ...]
            measurement_list_te.sort() # small to large
            partitions_list = []
            for single_measurement in measurement_list_te:
                partitions_list.append(np.around(single_measurement-NC_quantile, decimals=5))
                partitions_list.append(np.around(single_measurement+NC_quantile, decimals=5))
            partitions_list = list(np.unique(partitions_list)) #default is sorted
            for ind_candidate_intervals in range(len(partitions_list)+1):
                if ind_candidate_intervals == 0:
                    right_bar = partitions_list[ind_candidate_intervals]
                    left_bar = right_bar-10 # 10: any positive value for numerical stability
                elif ind_candidate_intervals == len(partitions_list):
                    left_bar = partitions_list[ind_candidate_intervals-1]
                    right_bar = left_bar + 10
                else:
                    left_bar = partitions_list[ind_candidate_intervals-1]
                    right_bar = partitions_list[ind_candidate_intervals]
                mid_bar = (left_bar+right_bar)/2
                # now count the number
                # interval_for_checking
                interval_for_counting = [mid_bar-NC_quantile, mid_bar+NC_quantile]
                measurement_np_te = np.array(measurement_list_te)
                count = sum((measurement_np_te >= interval_for_counting[0].numpy()) * (measurement_np_te <= interval_for_counting[1].numpy()))
                if count >= K_nn:
                    curr_interval = [left_bar, right_bar]
                    ineff += (right_bar-left_bar)
                    union_of_intervals.append(curr_interval)
                else:
                    pass
            return union_of_intervals, torch.tensor(ineff)
        else:
            raise NotImplementedError
    
    @staticmethod
    def compute_NC(NC_mode, single_example_pair, pqc_model, M, sampler, curr_fixed_measurement=None):
        x, y = single_example_pair
        if curr_fixed_measurement is None:
            measurement_list = pqc_model.M_shots_measurement_single(x, M, sampler)
        else:
            measurement_list = curr_fixed_measurement
        if 'distance_aware' in NC_mode:
            if isinstance(M, int):
                y_hat = sum(measurement_list)/len(measurement_list)
                curr_NC = torch.abs(y-torch.tensor(y_hat))
            else:
                y_hat = measurement_list.detach()
                curr_NC = torch.abs(y-y_hat)
            return curr_NC, y_hat # curr_NC: shape: (1)
        elif 'distribution_aware' in NC_mode: # either 'distribution_aware' or 'distribution_aware_k' k being the number of nearest neighbors (see eq 19 in the paper)
            assert isinstance(M, int) 
            curr_NC_list_tmp = []
            for y_m in measurement_list:
                curr_NC_list_tmp.append(torch.abs(y-torch.tensor(y_m)))
            if 'distribution_aware' == NC_mode:  # k(M)=1 in the paper (see Sec. IV-D)
                curr_NC = min(curr_NC_list_tmp)   
            else: # e.g., k(M)=M^0.5 in the paper (see Sec. IV-D)
                K_nn = int(NC_mode[21:]) # if K_nn = 1 same as above, else: K_nn is the k(M) in the paper
                curr_NC = torch.topk(torch.tensor(curr_NC_list_tmp), K_nn, largest=False, sorted=True).values[-1]
                curr_NC = curr_NC.unsqueeze(dim=0)
            return curr_NC, measurement_list # curr_NC: shape: (1)
        else:
            raise NotImplementedError
        




