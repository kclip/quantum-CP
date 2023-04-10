import numpy as np
import torch
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import TwoLocal
import time
import qiskit
import random
from .angle_encodings import Angle_Encoding, Angle_Encoding_linear, Angle_Encoding_linear_without_weight

class PQC_multi_qbit_qiskit:
    def __init__(self, num_layers, num_qubits, observation_min_max = [-2, 2], if_pqc_angle_encoding_dnn=False, if_vanilla_angle_encoding=False, angle_encoding_mode_for_with_input=None):
        self.observation_min_max = observation_min_max
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.obs_min = observation_min_max[0]
        self.obs_max = observation_min_max[1]
        self.obs_delta = (self.obs_max - self.obs_min)/((2**num_qubits)-1)

        self.if_pqc_angle_encoding_dnn = if_pqc_angle_encoding_dnn
        self.if_vanilla_angle_encoding = if_vanilla_angle_encoding
        if self.if_pqc_angle_encoding_dnn:
            if angle_encoding_mode_for_with_input == 'learned_nonlinear':
                self.quantum_para_list = Angle_Encoding(num_layer_PQC=num_layers, num_qubits=num_qubits)
            elif angle_encoding_mode_for_with_input == 'learned_linear':
                self.quantum_para_list = Angle_Encoding_linear(num_layer_PQC=num_layers, num_qubits=num_qubits)
            elif angle_encoding_mode_for_with_input == 'vanilla':
                self.quantum_para_list = Angle_Encoding_linear_without_weight(num_layer_PQC=num_layers, num_qubits=num_qubits)
            else:
                raise NotImplementedError
        else:
            assert self.if_vanilla_angle_encoding is True
            self.quantum_para_list = [] # [ {dict for first layer; 'qubit_0_w'=..'qubit_1_w', .., 'qubit_num_qubit_w'...}, {dict for second layer}   ]
            for ind_layer in range(num_layers):
                curr_layer = {}
                for ind_qubit in range(self.num_qubits):
                    curr_layer['qubit_' + str(ind_qubit) + '_weight'] = torch.rand(3, requires_grad=True)
                    curr_layer['qubit_' + str(ind_qubit) + '_bias'] = torch.rand(3, requires_grad=True)
                self.quantum_para_list.append(curr_layer)
        ## define measurement matrix
        self.observable_dict = {}
        for ind_observable in range(2**self.num_qubits):
            self.observable_dict[ind_observable] = self.obs_min + self.obs_delta*ind_observable

        self.service = QiskitRuntimeService()
        self.ansatz = TwoLocal(self.num_qubits, ['rz','ry','rz'], 'cz', 'linear', reps=self.num_layers-1, insert_barriers=True) 
        self.ansatz.measure_all()
        self.quantum_hardware = True

    def save_pqc(self, path):
        saving_info = {}
        saving_info['observation_min_max'] = self.observation_min_max
        saving_info['num_layers']  = self.num_layers
        saving_info['num_qubits']  = self.num_qubits
        saving_info['quantum_para_list']  = self.quantum_para_list
        saving_info['observable_dict']  = self.observable_dict
        torch.save(saving_info, path)
    def load_pqc(self, path):
        saved_info = torch.load(path)
        self.observation_min_max = saved_info['observation_min_max']
        self.num_layers = saved_info['num_layers']
        self.num_qubits = saved_info['num_qubits']
        self.quantum_para_list = saved_info['quantum_para_list']
        self.observable_dict = saved_info['observable_dict']
                
    def quantum_embedding(self, x, ps_info=None):
        assert ps_info is None
        # default zero state
        ### now transform with unitary matrices
        quantum_embedding_angle_dict = {}
        if self.if_vanilla_angle_encoding:
            for ind_layer in range(self.num_layers):
                curr_layer_para_dict = self.quantum_para_list[ind_layer]
                for ind_qubit in range(self.num_qubits):
                    curr_w = curr_layer_para_dict['qubit_' + str(ind_qubit) + '_weight']
                    curr_b = curr_layer_para_dict['qubit_' + str(ind_qubit) + '_bias']
                    curr_phi = curr_w * x + curr_b
                    quantum_embedding_angle_dict['ind_layer_'+ str(ind_layer) + 'ind_qubit_' + str(ind_qubit)] = curr_phi 
        else:
            curr_phi_concat = self.quantum_para_list(x)
            for ind_layer in range(self.num_layers):
                curr_layer_total_unitary = torch.tensor(1.0, dtype=torch.cfloat)
                for ind_qubit in range(self.num_qubits):
                    start_ind = ind_layer*self.num_qubits*3 + ind_qubit*3 
                    curr_phi = curr_phi_concat[ start_ind : start_ind+3 ]
                    quantum_embedding_angle_dict['ind_layer_'+ str(ind_layer) + 'ind_qubit_' + str(ind_qubit)] = curr_phi 

        quantum_embedding_angle_list = []
        for ind_layer in range(self.num_layers):
            for ind_phase in range(3):
                ind_phase_actual = (ind_phase+2)%3
                for ind_qubit in range(self.num_qubits):
                    ind_qubit_actual = self.num_qubits-1-ind_qubit
                    quantum_embedding_angle_list.append(quantum_embedding_angle_dict['ind_layer_'+ str(ind_layer) + 'ind_qubit_' + str(ind_qubit_actual)].type(torch.float64).detach().numpy()[ind_phase_actual]  )
        return quantum_embedding_angle_list

    def M_shots_measurement(self, x_list, M, sampler, ps_info=None):
        # x : [x_1, x_2, ...., x_N] # N: number of samples
        if ps_info is None:
            ps_info_list = [None] * len(x_list)
        else:
            ps_info_list = ps_info
        quantum_embedding_angle_list_total = []
        ind_x = 0
        for x in x_list:
            curr_quantum_embedding_angle_list = self.quantum_embedding(x, ps_info_list[ind_x])
            quantum_embedding_angle_list_total.append(curr_quantum_embedding_angle_list)
            ind_x += 1
        job = sampler.run(circuits=[self.ansatz]*len(x_list), parameter_values=quantum_embedding_angle_list_total, shots=M)
        result_batch = job.result()
        total_measurement_list = self.change_quasi_prob_to_measurements_batch(self.observable_dict, result_batch.quasi_dists, M)
        return total_measurement_list
    
    @staticmethod
    def change_quasi_prob_to_measurements_batch(observable_dict, results_quasi_dists_list, M):
        total_measurement_list = []
        for curr_quasi_dists in results_quasi_dists_list:
            # observable_dict: {'00000': 0.01 ...}
            curr_measurement_list = [] #.quasi_dists[0]['11']
            curr_candidate_for_addition_dict = {} # [ obs_0: 0.08, obs_1: 0.001, obs_2: 0.98, .... ]
            for eigen_vec in curr_quasi_dists.keys():
                curr_obs = observable_dict[eigen_vec]
                num_curr_obs = int(curr_quasi_dists[eigen_vec]*M)
                curr_candidate_for_addition_dict[curr_obs] = curr_quasi_dists[eigen_vec]*M - num_curr_obs # 0.xxx
                curr_measurement_list += [curr_obs]*num_curr_obs
            # now make the size same as M
            sorted_dict = {k: v for k, v in sorted(curr_candidate_for_addition_dict.items(), key=lambda item: item[1], reverse=True)}
            ind_add = 0
            while len(curr_measurement_list) < M:
                added_obs = list(sorted_dict.keys())[ind_add]
                curr_measurement_list.append(added_obs)
                ind_add += 1
            # now append to total list
            random.shuffle(curr_measurement_list)
            total_measurement_list.append(curr_measurement_list)
        return total_measurement_list
        