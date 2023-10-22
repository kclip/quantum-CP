import numpy as np

def PGM(rho_list, prob_y_list):
    S = 0
    for ind_class in range(len(prob_y_list)):
        S += rho_list[ind_class]*prob_y_list[ind_class]
    S_sqrt_inv = np.linalg.pinv(scipy.linalg.sqrtm(S))
    pgm_list = []
    for ind_class in range(len(prob_y_list)):
        pgm = S_sqrt_inv@prob_y_list[ind_class]*rho_list[ind_class]@S_sqrt_inv
        pgm_list.append(pgm)