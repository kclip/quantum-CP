import numpy as np
import scipy
from funcs.basic_funcs import sampling_from_prob_vec


class Finite_Shot_Pred:
    def __init__(self, rho_list, prob_y_list):
        self.rho_list = rho_list
        self.prob_y_list = prob_y_list
        self.pgm_list = self.PGM(rho_list, prob_y_list)
        self.num_classes = len(self.prob_y_list)

    def forward_fixed_rho(self, test_rho):
        likelihood_vec = np.zeros(self.num_classes)
        for ind_class in range(self.num_classes):
            likelihood_vec[ind_class] = np.round(np.trace(test_rho@self.pgm_list[ind_class]).real, decimals=8)
        likelihood_vec = likelihood_vec/np.sum(likelihood_vec) # numerical stability
        ind_prob = sampling_from_prob_vec(likelihood_vec)
        return ind_prob

    def forward(self, test_rho_pure, M, mode_CP, mode_drift, eps_drift):
        N = test_rho_pure.shape[0]
        histogram = np.zeros(self.num_classes)
        for m in range(M):
            if mode_drift == 'decoherence':
                test_rho = np.exp(-eps_drift*(m+1))*test_rho_pure + (1-np.exp(-eps_drift*(m+1)))*np.eye(N)/N
            else:
                test_rho = test_rho_pure
            ind_prob = self.forward_fixed_rho(test_rho)
            if mode_CP == 'poly_weighted_histo':
                histogram[ind_prob] += (1/M) * 1/(m+1)
            elif 'expo_weighted_histo' in mode_CP:
                multiplicative_factor = float(mode_CP[20:])
                histogram[ind_prob] += (1/M) * np.exp(-multiplicative_factor*eps_drift*(m+1)) #* 1/(m+1)
            else:
                histogram[ind_prob] += 1/M
        return histogram


    def forward_prev(self, test_rho, M):
        likelihood_vec = np.zeros(self.num_classes)
        for ind_class in range(self.num_classes):
            likelihood_vec[ind_class] = np.round(np.trace(test_rho@self.pgm_list[ind_class]).real, decimals=8)
        likelihood_vec = likelihood_vec/np.sum(likelihood_vec) # numerical stability
        histogram = np.zeros(self.num_classes)
        if M == 'infty':
            return likelihood_vec
        else:
            for m in range(M):
                ind_prob = sampling_from_prob_vec(likelihood_vec)
                histogram[ind_prob] += 1/M
            return histogram

    @staticmethod
    def PGM(rho_list, prob_y_list):
        S = 0
        for ind_class in range(len(prob_y_list)):
            S += rho_list[ind_class]*prob_y_list[ind_class]
        S_sqrt_inv = np.linalg.pinv(scipy.linalg.sqrtm(S))
        pgm_list = []
        for ind_class in range(len(prob_y_list)):
            pgm = prob_y_list[ind_class]*S_sqrt_inv@rho_list[ind_class]@S_sqrt_inv
            pgm_list.append(pgm)
        return pgm_list