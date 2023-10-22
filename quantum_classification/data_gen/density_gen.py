import qutip
import numpy as np
from scipy.linalg import logm, expm

class Data_Generator:
    def __init__(self, num_classes, num_dim):
        self.num_classes = num_classes
        self.num_dim = num_dim
    def forward(self, H_list, temperature, density=0.8):
        if H_list is None:
            H_list = []
            for ind_class in range(self.num_classes):
                rho = qutip.random_objects.rand_dm(self.num_dim, density)
                print(rho.shape)
                print('direct rho', np.sum(np.absolute(rho.full())<0.01)/100)
                H = self.rho_to_H(rho)
                H_list.append(H)
        else:
            pass
        rho_list = []
        for ind_class in range(self.num_classes):
            H = H_list[ind_class]
            tmp = expm(-H/temperature)
            print('temperature', temperature, 'Z', np.trace(tmp).real)
            rho = tmp/np.trace(tmp).real
            rho_list.append(rho)
        return H_list, rho_list

    @staticmethod
    def rho_to_H(rho):
        H = -logm(rho)
        return H

def Herm(A):
    return np.conj(A.T)


if __name__ == '__main__':
    DG = Data_Generator(10, 10)
    H_list, rho_list = DG.forward(None, 1.0, density = 0.8)
    print(np.sum(np.absolute(np.array(H_list))<0.01)/(10*100), np.sum(np.absolute(np.array(rho_list))<0.01)/(10*100))