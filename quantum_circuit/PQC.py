import torch
import torch.nn as nn
import numpy as np
from .angle_encodings import Angle_Encoding, Angle_Encoding_linear, Angle_Encoding_linear_without_weight

class PQC_multi_qbit:
    def __init__(self, num_layers, num_qubits, observation_min_max = [-2, 2], if_pqc_angle_encoding_dnn=False, if_vanilla_angle_encoding=False, angle_encoding_mode_for_with_input=None):
        self.observation_min_max = observation_min_max
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        # assuming CZ
        self.CZ_entangle = torch.zeros(4,4,dtype=torch.cfloat)
        self.CZ_entangle[0,0] = 1
        self.CZ_entangle[1,1] = 1
        self.CZ_entangle[2,2] = 1
        self.CZ_entangle[3,3] = -1
        # total entanglement
        self.total_entangle_matrix = self.compute_entanglement_matrix(num_qubits, self.CZ_entangle)
        self.obs_min = observation_min_max[0]
        self.obs_max = observation_min_max[1]
        self.obs_delta = (self.obs_max - self.obs_min)/((2**num_qubits)-1)
        self.if_pqc_angle_encoding_dnn = if_pqc_angle_encoding_dnn
        self.if_vanilla_angle_encoding = if_vanilla_angle_encoding
        if self.if_pqc_angle_encoding_dnn:
            ## this is for with input case
            if angle_encoding_mode_for_with_input == 'learned_nonlinear':
                self.quantum_para_list = Angle_Encoding(num_layer_PQC=num_layers, num_qubits=num_qubits)
            elif angle_encoding_mode_for_with_input == 'learned_linear':
                self.quantum_para_list = Angle_Encoding_linear(num_layer_PQC=num_layers, num_qubits=num_qubits)
            elif angle_encoding_mode_for_with_input == 'fixed':
                self.quantum_para_list = Angle_Encoding_linear_without_weight(num_layer_PQC=num_layers, num_qubits=num_qubits)
            else:
                raise NotImplementedError
        else:
            if self.if_vanilla_angle_encoding: # essentially same as 'fixed' case above
                self.quantum_para_list = [] # [ {dict for first layer; 'qubit_0_w'=..'qubit_1_w', .., 'qubit_num_qubit_w'...}, {dict for second layer}   ]
                for ind_layer in range(num_layers):
                    curr_layer = {}
                    for ind_qubit in range(self.num_qubits):
                        curr_layer['qubit_' + str(ind_qubit) + '_weight'] = torch.zeros(1, requires_grad=False) #torch.rand(3, requires_grad=True)
                        curr_layer['qubit_' + str(ind_qubit) + '_bias'] = torch.rand(3, requires_grad=True)
                    self.quantum_para_list.append(curr_layer)
            else:
                raise NotImplementedError

        self.projection_matrix_dict = {}
        self.observable_dict = {}
        for ind_projection_matrix in range(2**self.num_qubits):
            curr_one_hot = torch.zeros(2**self.num_qubits, dtype=torch.cfloat)
            curr_one_hot[ind_projection_matrix] = 1.0 # orthonormal vector
            curr_one_hot = curr_one_hot.unsqueeze(dim=1)# (2^n, 1) vector
            self.projection_matrix_dict[ind_projection_matrix] = curr_one_hot @ self.Herm(curr_one_hot)
            self.observable_dict[ind_projection_matrix] = self.obs_min + self.obs_delta*ind_projection_matrix
        self.quantum_hardware = False

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
        if ps_info is None:
            pass
        else:
            ind_shifted_layer, ind_shifted_qubit, ind_shifted_phi, p_m_mode = ps_info # ind_shifted_phi: 0,1,2
        init_single_qubit = torch.zeros(2, dtype=torch.cfloat)
        init_single_qubit[0] = 1 # ([1,0])
        init_single_qubit = init_single_qubit.unsqueeze(dim=1) # (2,1)
        qubits = torch.tensor(1.0, dtype=torch.cfloat)
        for ind_qubit in range(self.num_qubits):
            qubits = torch.kron(qubits, init_single_qubit)

        if not self.if_vanilla_angle_encoding: # supervised learning case that considers multiple angle encoding approaches (Fig. 9 in the paper)
            curr_phi_concat = self.quantum_para_list(x)
            for ind_layer in range(self.num_layers):
                curr_layer_total_unitary = torch.tensor(1.0, dtype=torch.cfloat)
                for ind_qubit in range(self.num_qubits):
                    start_ind = ind_layer*self.num_qubits*3 + ind_qubit*3 
                    curr_phi = curr_phi_concat[ start_ind : start_ind+3 ]
                    if ps_info is None:
                        pass
                    else:
                        # ps_info can be useful when one implements PS rule to update the PQC! we implement this functionality hoping that it may be useful to some readers
                        if (ind_layer == ind_shifted_layer) and (ind_qubit == ind_shifted_qubit):
                            if p_m_mode == '+':
                                curr_phi[ind_shifted_phi] += np.pi/2
                            elif p_m_mode == '-':
                                curr_phi[ind_shifted_phi] -= np.pi/2
                            else:
                                raise NotImplementedError
                        else:
                            pass
                    curr_unitary = self.rot_z(curr_phi[1])@self.rot_y(curr_phi[0])@self.rot_z(curr_phi[2])
                    ### this is for a single qubit, we need to kron for every qubits!
                    curr_layer_total_unitary = torch.kron(curr_layer_total_unitary, curr_unitary)
                qubits = curr_layer_total_unitary @ qubits # (2^n, 1)
                #### now apply entanglement
                if ind_layer == self.num_layers -1:
                    pass # we assume no entanglement at the last layer
                else:
                    qubits = self.total_entangle_matrix @ qubits
        else: ### unsupervised learning uses this for simplicity # Fig. 7 in the paper
            for ind_layer in range(self.num_layers):
                curr_layer_para_dict = self.quantum_para_list[ind_layer]
                curr_layer_total_unitary = torch.tensor(1.0, dtype=torch.cfloat)
                for ind_qubit in range(self.num_qubits):
                    curr_w = curr_layer_para_dict['qubit_' + str(ind_qubit) + '_weight']
                    curr_b = curr_layer_para_dict['qubit_' + str(ind_qubit) + '_bias']
                    curr_phi = curr_w * x + curr_b
                    if ps_info is None:
                        pass
                    else:
                        # ps_info can be useful when one implements PS rule to update the PQC! we implement this functionality hoping that it may be useful to some readers
                        if (ind_layer == ind_shifted_layer) and (ind_qubit == ind_shifted_qubit):
                            if p_m_mode == '+':
                                curr_phi[ind_shifted_phi] += np.pi/2
                            elif p_m_mode == '-':
                                curr_phi[ind_shifted_phi] -= np.pi/2
                            else:
                                raise NotImplementedError
                        else:
                            pass
                    curr_unitary = self.rot_z(curr_phi[1])@self.rot_y(curr_phi[0])@self.rot_z(curr_phi[2])
                    ### this is for a single qubit, we need to kron for every qubits!
                    curr_layer_total_unitary = torch.kron(curr_layer_total_unitary, curr_unitary)
                qubits = curr_layer_total_unitary @ qubits # (2^n, 1)
                #### now apply entanglement
                if ind_layer == self.num_layers -1:
                    pass # no entanglement at the last layer!
                else:
                    qubits = self.total_entangle_matrix @ qubits
        return qubits

    def M_shots_measurement(self, x_list, M, sampler, ps_info=None):
        if ps_info is None:
            ps_info_list = [None] * len(x_list)
        else:
            ps_info_list = ps_info
        total_measurement_list = []
        ind_x = 0
        for x in x_list:
            curr_measurement_list = self.M_shots_measurement_single(x, M, sampler, ps_info=ps_info_list[ind_x])
            total_measurement_list.append(curr_measurement_list)
            ind_x += 1
        return total_measurement_list


    def M_shots_measurement_with_multiple_parameters(self, x_list, M, sampler, list_para_list, ps_info=None):
        # x : [x_1, x_2, ...., x_N] # N: number of samples
        total_measurement_list = []
        ind_x = 0
        for x in x_list:
            self.quantum_para_list = list_para_list[ind_x]
            curr_measurement_list = self.M_shots_measurement_single(x, M, sampler, ps_info=None)
            total_measurement_list.append(curr_measurement_list)
            ind_x += 1
        return total_measurement_list
    

    def M_shots_measurement_single(self, x, M, sampler, ps_info=None): # sampler for qiskit quantum computer, used here for consistency
        assert sampler is None # we are doing simulation, no need for qiskit at all!
        embedded_qubit_by_x = self.quantum_embedding(x, ps_info)
        # M = number or 'infty_for_mean' or 'infty_for_prob'
        # embedded_qubit_by_x: (2^n, 1)
        embedded_density_matrix  = embedded_qubit_by_x @ self.Herm(embedded_qubit_by_x)
        #print('embedded_density_matrix', embedded_density_matrix)
        if (M=='infty_for_mean') or (M=='infty_for_prob'):
            # usual mean with gt prob
            # compute gt prob for each elem
            prob_value_list = [] # (2^n elements)
            mean_regression = 0
            for ind_projection_matrix in self.observable_dict.keys(): # 0~ 2^n-1
                curr_prob = torch.trace(embedded_density_matrix @ self.projection_matrix_dict[ind_projection_matrix])
                assert curr_prob.imag < 0.0000001
                prob_value_list.append(curr_prob.real)
                if 'mean' in M:
                    mean_regression += curr_prob.real*self.observable_dict[ind_projection_matrix]
                else:
                    pass
            if 'mean' in M:
                return mean_regression
            else:
                return prob_value_list
        else:
            prob_value_list = [] # (2^n elements)
            for ind_projection_matrix in self.observable_dict.keys(): # 0~ 2^n-1
                curr_prob = torch.trace(embedded_density_matrix @ self.projection_matrix_dict[ind_projection_matrix])
                assert curr_prob.imag < 0.0000001
                prob_value_list.append(curr_prob.real.unsqueeze(dim=0))
            prob_value_vec = torch.cat(prob_value_list, dim=0)
            cumsum_prob = torch.cumsum(prob_value_vec, dim=0) # (2^n, 1)
            measurement_list = []
            for m in range(M):
                rand_sample = torch.rand(1) # [0, 1)
                for ind_prob in range(len(cumsum_prob)):
                    if ind_prob == 0:
                        if 0 <= rand_sample < cumsum_prob[ind_prob]:
                            curr_measurement = self.observable_dict[ind_prob]
                            break
                    else:
                        if cumsum_prob[ind_prob-1] <= rand_sample < cumsum_prob[ind_prob]:
                            curr_measurement = self.observable_dict[ind_prob]
                            break
                if rand_sample >= cumsum_prob[-1]:
                    assert ind_prob == len(cumsum_prob) - 1
                    curr_measurement = self.observable_dict[ind_prob]
                else:
                    pass
                measurement_list.append(curr_measurement)
            return measurement_list
      

    @staticmethod
    def rot_z(phi):
        rot_matrix = torch.zeros(2,2,dtype=torch.cfloat)
        rot_matrix[0,0] = torch.exp(1j*phi/2)
        rot_matrix[1,1] = torch.exp(-1j*phi/2)
        return rot_matrix
    @staticmethod
    def rot_y(phi):
        rot_matrix = torch.zeros(2,2,dtype=torch.cfloat)
        rot_matrix[0,0] = torch.cos(phi/2)
        rot_matrix[0,1] = torch.sin(phi/2)
        rot_matrix[1,0] = -torch.sin(phi/2)
        rot_matrix[1,1] = torch.cos(phi/2)
        return rot_matrix
    @staticmethod 
    def compute_curr_two_qubits_entanglement_matrix(basic_entanglement, num_qubits, ind_first_qubit):
        # basic_entanglement : (4,4) CZ gate
        curr_entire_entangle = torch.tensor(1.0, dtype=torch.cfloat)
        # among index 0, ..., num_qubits-1 -> ind_first_qubit, ind_first_qubit+1 are CZ while others are identity
        for ind_qubit in range(num_qubits):
            if ind_qubit == ind_first_qubit:
                curr_entire_entangle = torch.kron(curr_entire_entangle, basic_entanglement)
            elif ind_qubit == ind_first_qubit+1:
                pass # already multiplied (4,4) matrix
            else:
                curr_entire_entangle = torch.kron(curr_entire_entangle, torch.eye(2))
        return curr_entire_entangle # (2^n, 2^n)
            
    def compute_entanglement_matrix(self, num_qubits, basic_entanglement):
        total_entangle = torch.tensor(1.0, dtype=torch.cfloat)
        for ind_entangle in range(num_qubits-1):
            curr_entangle = self.compute_curr_two_qubits_entanglement_matrix(basic_entanglement, num_qubits, ind_entangle)
            if ind_entangle == 0:
                total_entangle = curr_entangle
            else:
                total_entangle @= curr_entangle
        return total_entangle # (2^n, 2^n)
    
    @staticmethod
    def compute_total_rot_matrix_per_layer(curr_layer_phi_list):
        #curr_layer_phi_list = [phi for 1st qubit, phi for 2nd qubit ....]
        total_rot_matrix = torch.tensor(1.0, dtype=torch.cfloat)
        for curr_phi in curr_layer_phi_list:
            curr_unitary = self.rot_z(curr_phi[1])@self.rot_y(curr_phi[0])@self.rot_z(curr_phi[2])
            total_rot_matrix = torch.kron(total_rot_matrix, curr_unitary)
        return total_rot_matrix # (2^n, 2^n)
    @staticmethod
    def Herm(vector):
        return torch.transpose(torch.conj(vector),0,1)
            
