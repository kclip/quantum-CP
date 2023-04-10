import torch
from scipy.stats import norm

class Bimodal_Generator_No_Input:
    def __init__(self, dummy_x_input = 0.0, y_mean_list = [-0.75, 0.75], std=0.1):
        self.dummy_x_input = torch.tensor(dummy_x_input)
        self.dim_x = 1
        self.y_mean_list = y_mean_list
        self.std = std
    def gen(self, num_examples):
        X = []
        Y = []
        for ind_sample in range(num_examples):
            y = self.multimodal_without_x(self.y_mean_list, self.std, ind_sample)
            X.append(self.dummy_x_input.unsqueeze(dim=0)) # (1,dim_x)
            Y.append(y.unsqueeze(dim=0)) # (1,1)
        X = torch.cat(X, dim=0) # (num_samples, dim_x)
        Y = torch.cat(Y, dim=0) # (num_samples, 1)
        if self.dim_x == 1:
            X = X.unsqueeze(dim=1) # (num_samples, dim_x=1)
        else:
            pass
        Y = Y.unsqueeze(dim=1)  # (num_samples, 1)
        return [X, Y]   

    def get_entire_data_set(self, num_indep_experiments, num_tr_data_examples, num_val_data_examples, num_te_data_examples):
        total_dataset_over_indep_experiments = {}
        for ind_indep_exp in range(num_indep_experiments): 
            gt_dataset_dict = {}
            tr_set_genie = self.gen(num_tr_data_examples)
            gt_dataset_dict['tr'] = tr_set_genie
            gt_dataset_dict['val'] = self.gen(num_val_data_examples)
            te_exact_coverage_computation = self.gen(1)
            te_exact_coverage_computation[1] = [self] # we will use analytical integration to average over the true target distribution for unsupervised learning case
            gt_dataset_dict['te'] = te_exact_coverage_computation
            curr_entire_dataset = [gt_dataset_dict]
            total_dataset_over_indep_experiments['ind_indep_exp_' + str(ind_indep_exp)] = curr_entire_dataset
        return total_dataset_over_indep_experiments
    
    @staticmethod
    def multimodal_without_x(y_mean_list, std, ind_sample):
        assert len(y_mean_list) == 2
        if torch.rand(1) < 0.5:
            y = torch.normal(torch.tensor(y_mean_list[0]), std)
        else:
            y = torch.normal(torch.tensor(y_mean_list[1]), std)
        return y

    def compute_coverage_for_single_interval(self, interval):
        first_mode_cov = 0.5*(norm.cdf(interval[1], loc=self.y_mean_list[0], scale=self.std) - norm.cdf(interval[0], loc=self.y_mean_list[0], scale=self.std))
        second_mode_cov = 0.5*(norm.cdf(interval[1], loc=self.y_mean_list[1], scale=self.std) - norm.cdf(interval[0], loc=self.y_mean_list[1], scale=self.std))
        return first_mode_cov + second_mode_cov
    