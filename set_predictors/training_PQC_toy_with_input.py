import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from funcs.loss_computation import Loss_compute
import platform
import time
import copy

class PQC_Trainer_with_input:
    def __init__(self, pqc_model, M_tr, training_mode, tr_dataset, num_tr_iter, lr, angle_encoding_mode_for_with_input=None):
        self.pqc_model = pqc_model
        self.M_tr = M_tr
        self.training_mode = training_mode
        self.tr_dataset = tr_dataset
        self.num_tr_iter = num_tr_iter
        self.lr = lr
        self.angle_encoding_mode_for_with_input = angle_encoding_mode_for_with_input

    def forward(self, sampler=None, mb_size=16): 
        X_tr = self.tr_dataset[0]
        Y_tr = self.tr_dataset[1]
        num_tr = X_tr.shape[0]
        print('M_tr: ', self.M_tr, 'num_tr: ', num_tr)
        num_layers = self.pqc_model.num_layers
        num_qubits = self.pqc_model.num_qubits
        if mb_size is None:
            mb_size = num_tr # GD
        else:
            pass
        if num_tr < mb_size:
            mb_size = num_tr // 2 
        if mb_size == 0:
            mb_size = 1
        all_possible_y_list = list(self.pqc_model.observable_dict.values()) # this is fixed
        iter_list = []
        loss_per_iter = []
        loss_computer = Loss_compute(mode=self.training_mode)
        divergence_info = (None, None, None)

        if isinstance(self.pqc_model.quantum_para_list, list):
            pass
        else:
            optimizer = torch.optim.Adam(self.pqc_model.quantum_para_list.parameters(), self.lr)
        loss_list = []

        for iter in range(self.num_tr_iter):
            if isinstance(self.pqc_model.quantum_para_list, list):
                pass
            else:   
                optimizer.zero_grad()
            mse_loss = 0
            div_loss = 0
            curr_perm = torch.randperm(num_tr)
            total_grad_list = []
            for ind_mb in range(mb_size):
                curr_idx_tr = curr_perm[ind_mb] 
                x = X_tr[curr_idx_tr] # size: (1)
                y_rep_per_x = Y_tr[curr_idx_tr] # size: (y_rep)
                target_distribution_info = {}
                K = y_rep_per_x.shape[0]
                if self.M_tr == 'infty_for_prob':
                    for y in y_rep_per_x:
                        # first find neareast neighbor
                        delta_y_cand = 999999
                        for y_cand in all_possible_y_list:
                            curr_delta = np.abs(y_cand-y)
                            if curr_delta<delta_y_cand:
                                best_y = y_cand
                                delta_y_cand = curr_delta
                            else:
                                pass
                        if best_y in target_distribution_info.keys():
                            target_distribution_info[best_y] += torch.tensor(1/K)
                        else:
                            target_distribution_info[best_y] = torch.tensor(1/K)
                    measurements = self.pqc_model.M_shots_measurement_single(x, self.M_tr, sampler)
                    source_distribution_info = {}
                    ind_y = 0
                    for y in all_possible_y_list:
                        source_distribution_info[y] = measurements[ind_y]
                        ind_y += 1 
                    div_loss += loss_computer.compute(source_distribution_info, target_distribution_info)  
                    measurement_list_from_no_shift_list = [measurements]
                elif self.M_tr == 'infty_for_mean':
                    ### quadratic loss
                    mean = self.pqc_model.M_shots_measurement_single(x, 'infty_for_mean', sampler)
                    mse_loss += torch.sum((mean-y_rep_per_x)**2)                        
                    measurement_list_from_no_shift_list = [[mean]]
                else:
                    print('in order to account for finite number of measurements during training, we need to use PS rule or some gradient-free optimization, which we did not consider in this work')
                    raise NotImplementedError
            if self.M_tr == 'infty_for_prob':
                loss = div_loss
            elif self.M_tr == 'infty_for_mean':
                loss = mse_loss
            else:
                raise NotImplementedError
            loss_list.append(float(loss/mb_size))
            if isinstance(self.pqc_model.quantum_para_list, list):
                for ind_layer in range(num_layers):
                    for ind_qubit in range(num_qubits):
                        for w_b in ['_weight', '_bias']: #['_bias']: #
                            if self.M_tr == 'infty_for_prob':
                                curr_grad = torch.autograd.grad(loss/mb_size, self.pqc_model.quantum_para_list[ind_layer]['qubit_' + str(ind_qubit) + w_b], retain_graph=True)[0]
                            else:
                                curr_grad = total_grad_list[ind_layer]['qubit_' + str(ind_qubit) +w_b]
                            self.pqc_model.quantum_para_list[ind_layer]['qubit_' + str(ind_qubit) + w_b] = self.pqc_model.quantum_para_list[ind_layer]['qubit_' + str(ind_qubit) + w_b] - self.lr * curr_grad
            else:
                loss_for_backprop = loss/mb_size
                loss_for_backprop.backward()
                optimizer.step()


def change_measurements_to_quasi_prob(measurements, all_possible_y_list):
    source_distribution_info = {}
    K = len(measurements)
    for single_shot in measurements:
        if single_shot in source_distribution_info.keys():
            source_distribution_info[single_shot] += torch.tensor(1/K)
        else:
            source_distribution_info[single_shot] = torch.tensor(1/K)
    return source_distribution_info
