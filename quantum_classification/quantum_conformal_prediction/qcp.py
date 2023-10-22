import numpy as np
from pretty_good_measurement.finite_shot_pred import Finite_Shot_Pred
from funcs.basic_funcs import sampling_from_prob_vec

MAX_VAL = 999999999999

class QCP:
    def __init__(self, alpha, M, num_cal, rho_list, prob_y_list, mode_CP, mode_drift, eps_drift):
        self.num_cal = num_cal
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
            threshold = self.quantile_using_cal()
            for ind_test in range(num_test_per_rep):
                ind_class_test = sampling_from_prob_vec(np.array(self.prob_y_list))
                test_rho = self.rho_list[ind_class_test]
                predicted_labels = self.set_pred(test_rho, threshold)
                coverage, ineff = self.compute_cov_ineff(predicted_labels, ind_class_test)
                cov_list.append(coverage)
                ineff_list.append(ineff)
        cov_vec = np.array(cov_list)
        ineff_vec = np.array(ineff_list)
        return (np.mean(cov_vec), np.var(cov_vec, ddof=1)), (np.mean(ineff_vec), np.var(ineff_vec, ddof=1))

    def quantile_using_cal(self):
        score_list = []
        for ind_cal in range(self.num_cal):
            ## gen cal set
            ind_class_cal = sampling_from_prob_vec(np.array(self.prob_y_list))
            cal_rho = self.rho_list[ind_class_cal]
            ## get score
            histogram = self.PGM_pred.forward(cal_rho, self.M, self.mode_CP, self.mode_drift, self.eps_drift)
            if histogram[ind_class_cal] == 0:
                score = MAX_VAL
            else:
                score = 1/histogram[ind_class_cal]
            score_list.append(score)
        sorted_score = np.sort(np.array(score_list))
        index_quan = int(np.ceil((1-self.alpha)*(self.num_cal+1)))-1 # python start indexing from 0!
        if index_quan >= self.num_cal:
            print('not enough cal! so returning trivial set (entire set)')
            return MAX_VAL*10
        else:
            pass
        return sorted_score[index_quan]

    def set_pred(self, test_rho, threshold):
        histogram = self.PGM_pred.forward(test_rho, self.M, self.mode_CP, self.mode_drift, self.eps_drift)
        predicted_labels = []
        for ind_class in range(self.num_classes):
            if histogram[ind_class] == 0:
                curr_score = MAX_VAL
            else:
                curr_score = 1/histogram[ind_class]
            if curr_score <= threshold:
                predicted_labels.append(ind_class)
            else:
                pass
        return predicted_labels

    def compute_cov_ineff(self, predicted_labels, ind_class_test):
        ineff = len(predicted_labels)
        if ind_class_test in predicted_labels:
            coverage = 1
        else:
            coverage = 0
        return coverage, ineff