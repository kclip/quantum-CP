import numpy as np
from data_gen.density_gen import Data_Generator
from pretty_good_measurement.finite_shot_pred import Finite_Shot_Pred
import matplotlib.pyplot as plt
import os
import pickle
from quantum_conformal_prediction.qcp import QCP
from quantum_conformal_prediction.naive import Naive_Set_Pred
import scipy.io as sio
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='QCP_QC')
    parser.add_argument('--temperature', type=float, default=0.01, help='temperature of the Gibbs state')
    parser.add_argument('--mode', type=str, default='naive', help='either naive or qcp')
    parser.add_argument('--mode_CP', type=str, default='vanilla', help='either weighted_histo or vanilla (weighted histo corresponds to QCP, vanialla corresponds to CP)')
    parser.add_argument('--mode_drift', type=str, default='decoherence', help='either pure, decoherence (pure: no drift, decoherence: with drift)')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()    
    print('args', args)
    num_classes = 10 # number of different density matrices to classify
    num_dim = 16 # dimension of the density matrix (num_dim * num_dim)
    temp_list = [args.temperature] 
    prob_y_list = [1/num_classes for i in range(num_classes)] # uniform label probability
    class_list = [i for i in range(num_classes)]
    M_list = [100] # number of measurement shots
    eps_drift_list = [0.0, 0.01, 0.02, 0.05,0.1, 0.2, 0.5,1.0, 2.0, 5.0, 10.0]   # eps_drift is defined as the inverse of tau in eq.(15) of the paper
    alpha = 0.1 # target miscoverage level
    num_cal = 10 # number of calibration data
    num_rep = 1000 # number of repetition 
    num_test_per_rep = 10 # number of test data per repetition

    # for H gen
    H_path = './saved_dataset/' + 'num_classes_' + str(num_classes) + '/H.p'
    if os.path.isfile(H_path):
        with open(H_path, 'rb') as fp:
            H_list = pickle.load(fp)
    else:
        Data_Gen = Data_Generator(num_classes, num_dim)
        H_list, rho_list = Data_Gen.forward(None, 1.0)
        os.makedirs('./saved_dataset/' + 'num_classes_' + str(num_classes))
        with open(H_path, 'wb') as fp:
            pickle.dump(H_list, fp)

    for eps_drift in eps_drift_list:
        args.eps_drift = eps_drift
        print('eps drift', eps_drift)
        for temperature in temp_list:
            rho_path = './saved_dataset/'  + 'num_classes_' + str(num_classes) + '/rho_list_temp' + str(temperature) + '.p'
            if os.path.isfile(rho_path):
                with open(rho_path, 'rb') as fp:
                    rho_list = pickle.load(fp)
            else:
                Data_Gen = Data_Generator(num_classes, num_dim)
                _, rho_list = Data_Gen.forward(H_list, temperature)
                with open(rho_path, 'wb') as fp:
                    pickle.dump(rho_list, fp)

        for temperature in temp_list:
            eval_dict = {}
            eval_dir = './saved_results/' +  'num_classes_' + str(num_classes) +  '/mode_' + args.mode + '/temp_' + str(temperature) + '/num_cal_'+str(num_cal) + '/mode_CP_' + str(args.mode_CP) + 'mode_drift_' + str(args.mode_drift) + 'eps_drift_' + str(args.eps_drift) + '/' + 'M_'+str(M_list[0]) + '_' + str(M_list[-1]) + '/'
            if os.path.isdir(eval_dir):
                pass
            else:
                os.makedirs(eval_dir)
            rho_path = './saved_dataset/'  + 'num_classes_' + str(num_classes) + '/rho_list_temp' + str(temperature) + '.p'
            with open(rho_path, 'rb') as fp:
                rho_list = pickle.load(fp)
            mean_cov_list = []
            mean_ineff_list = []
            var_cov_list = []
            var_ineff_list = []
            for M in M_list:
                if args.mode == 'qcp':
                    qcp = QCP(alpha, M, num_cal, rho_list, prob_y_list, args.mode_CP, args.mode_drift, args.eps_drift)
                    (cov_mean, cov_var), (ineff_mean, ineff_var) = qcp.forward(num_rep, num_test_per_rep)
                else:
                    naive = Naive_Set_Pred(alpha, M, rho_list, prob_y_list, args.mode_CP, args.mode_drift, args.eps_drift)
                    (cov_mean, cov_var), (ineff_mean, ineff_var) = naive.forward(num_rep, num_test_per_rep)
                print('temperature', temperature, 'M', M, 'avg cov', cov_mean, 'avg ineff', ineff_mean)
                mean_cov_list.append(cov_mean)
                mean_ineff_list.append(ineff_mean)
                var_cov_list.append(cov_var)
                var_ineff_list.append(ineff_var)
            eval_dict['cov_mean'] = mean_cov_list
            eval_dict['ineff_mean'] = mean_ineff_list
            eval_dict['cov_var'] = var_cov_list
            eval_dict['ineff_var'] = var_ineff_list
            sio.savemat(eval_dir + '/eval_results.mat', eval_dict)



            