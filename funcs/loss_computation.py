import torch
import numpy as np

class Loss_compute:
    def __init__(self, mode):
        # source_distribution_info: {'x': q(x)}
        # target: {'x': p(x)}
        if mode == 'forward_KL': # forward KL is same as the common cross-entropy loss (eq 41 in the paper)
            self.compute_divergece = KL('forward')
        else:
            raise NotImplementedError

    def compute(self, source_distribution_info, target_distribution_info):
        # source_distribution_info: what we want to optimize (PQC distribution)
        # target_distribution_info: gt distribution (can be replaced with empirical distribution)
        divergence, _ = self.compute_divergece.forward(source_distribution_info, target_distribution_info)
        return divergence
    

class KL:
    def __init__(self, kl_mode):
        self.kl_mode = kl_mode
    def forward(self, source_distribution_info, target_distribution_info):
        if self.kl_mode == 'forward':
            P = target_distribution_info
            Q = source_distribution_info
        elif self.kl_mode == 'backward':
            P = source_distribution_info
            Q = target_distribution_info
        else:
            raise NotImplementedError
        kl_div = 0
        for x in P.keys():
            if x in Q.keys():
                if torch.isnan(torch.log(P[x]/Q[x])):
                    pass
                elif torch.isinf(torch.log(P[x]/Q[x])):
                    pass
                else:
                    kl_div += P[x]*torch.log(P[x]/Q[x])
            else:
                pass
        return kl_div, None
        

