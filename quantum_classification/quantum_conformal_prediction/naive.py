import numpy as np
from pretty_good_measurement.finite_shot_pred import Finite_Shot_Pred
from funcs.basic_funcs import sampling_from_prob_vec

class Naive_Set_Pred:
    def __init__(self, alpha, M, rho_list, prob_y_list, mode_CP, mode_drift, eps_drift):
        self.rho_list = rho_list
        self.prob_y_list = prob_y_list
        self.PGM_pred = Finite_Shot_Pred(rho_list, prob_y_list)
        self.M = M
        self.alpha = alpha
        self.num_classes= len(prob_y_list)
        self.mode_CP = mode_CP
        self.mode_drift = mode_drift
        self.eps_drift = eps_drift
    def forward(self, num_rep, num_test_per_rep):
        cov_list = []
        ineff_list = []
        for ind_rep in range(num_rep):
            for ind_test in range(num_test_per_rep):
                ind_class_test = sampling_from_prob_vec(np.array(self.prob_y_list))
                test_rho = self.rho_list[ind_class_test]
                predicted_labels = self.set_pred(test_rho)
                coverage, ineff = self.compute_cov_ineff(predicted_labels, ind_class_test)
                cov_list.append(coverage)
                ineff_list.append(ineff)
        cov_vec = np.array(cov_list)
        ineff_vec = np.array(ineff_list)
        return (np.mean(cov_vec), np.var(cov_vec, ddof=1)), (np.mean(ineff_vec), np.var(ineff_vec, ddof=1))

    def set_pred(self, test_rho):
        histogram = self.PGM_pred.forward(test_rho, self.M, self.mode_CP, self.mode_drift, self.eps_drift)
        histogram /= np.sum(histogram) # ensuring probability measure
        sorted_histo = np.sort(histogram)
        argsorted_histo = np.argsort(histogram)
        predicted_labels = []
        cum_prob = 0
        ind_max = 1
        while cum_prob < 1-self.alpha:
            predicted_labels.append(argsorted_histo[-ind_max])
            cum_prob += sorted_histo[-ind_max]
            ind_max += 1
        return predicted_labels

    def compute_cov_ineff(self, predicted_labels, ind_class_test):
        ineff = len(predicted_labels)
        if ind_class_test in predicted_labels:
            coverage = 1
        else:
            coverage = 0
        return coverage, ineff