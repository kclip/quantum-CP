import torch
import numpy as np
from scipy.stats import norm

class Bimodal_Generator_With_Input:
    def __init__(self, x_scale = 1.0, std=0.05, x_min_max = [-10, 10]):
        self.dim_x = 1
        self.std = std
        self.x_min_max = x_min_max
        self.x_min_max[0] /= x_scale
        self.x_min_max[1] /= x_scale
        self.x_scale = x_scale
    def gen(self, num_examples):
        X = []
        Y = []
        for ind_sample in range(num_examples):
            x = torch.rand(1)*( self.x_min_max[1]-self.x_min_max[0] ) + self.x_min_max[0]
            y_per_x = torch.zeros(1, 1)
            y_per_x[:, 0] = self.multimodal_with_x(x, self.std)
            X.append(x*self.x_scale) # (1,dim_x)
            Y.append(y_per_x) # (1,1)
        if num_examples == 0:
            X = None
            Y = None
        else:
            X = torch.cat(X, dim=0) # (num_samples, dim_x)
            Y = torch.cat(Y, dim=0) # (num_samples, 1)
            if self.dim_x == 1:
                X = X.unsqueeze(dim=1) # (num_samples, dim_x=1)
            else:
                pass
        return [X, Y]

    def get_entire_data_set(self, num_indep_experiments, num_tr_data_examples, num_val_data_examples, num_te_data_examples):
        total_dataset_over_indep_experiments = {}
        for ind_indep_exp in range(num_indep_experiments): # 1000
            gt_dataset_dict = {}
            if ind_indep_exp == 0:
                tr_set_genie = self.gen(num_tr_data_examples)
                gt_dataset_dict['tr'] = tr_set_genie
            else:
                gt_dataset_dict['tr'] = None # only train once
            gt_dataset_dict['val'] = self.gen(num_val_data_examples) 
            gt_dataset_dict['te'] = self.gen(num_te_data_examples) 
            curr_entire_dataset = [gt_dataset_dict]
            total_dataset_over_indep_experiments['ind_indep_exp_' + str(ind_indep_exp)] = curr_entire_dataset
        return total_dataset_over_indep_experiments
    
    @staticmethod
    def compute_mean_given_x(x):
        return 0.5*np.sin(0.8*x.numpy()) + 0.05*x.numpy()

    def multimodal_with_x(self, x, std):
        mean = self.compute_mean_given_x(x)
        if torch.rand(1) < 0.5:
            coin_flip = 1
        else:
            coin_flip = -1
        y = torch.normal(coin_flip*torch.tensor(mean), std)
        return y
