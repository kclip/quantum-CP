import torch
import argparse
import numpy as np
from quantum_circuit.PQC import PQC_multi_qbit
from funcs.basic_utils import reset_random_seed
from funcs.utils_for_eval import cov_ineff_compute
from set_predictors.quantum_conformal_prediction import QCP
from set_predictors.quantum_naive_set_prediction_using_quantile import Naive_Predictor
from funcs.utils_for_main import data_gen_for_unsup_learning, nearest_neighbor_adaptive_to_M
from funcs.experimental_setting_unsup import automatic_args_setting_unsup, training_pqc_unsup, bring_saved_measurements

def parse_args():
    parser = argparse.ArgumentParser(description='QCP for unsupervised learning')
    parser.add_argument('--alpha', type=float, default=0.1, help='predetermined miscoverage level')
    parser.add_argument('--set_prediction_mode', type=str, default='CP', help='either naive_quan (for naive set predictor),  or CP')
    parser.add_argument('--NC_mode', type=str, default='distribution_aware_K_adaptive', help='scoring function, default is the proposed scoring function for QCP') 
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--exp_mode', type=str, default='toy_without_input', help='toy_without_input, toy_with_input') #'Blei_original_single_c', 'distance_aware'    
    parser.add_argument('--num_training_iter', type=int, default=50, help='training iterations')
    parser.add_argument('--lr', type=float, default=1, help='learning rate') 
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers (L) for PQC')    
    parser.add_argument('--if_fixed_measurement_for_quantum_run', dest='if_fixed_measurement_for_quantum_run', action='store_true', default=False, help='using fixed measurements from saved file')    
    parser.add_argument('--x_scale', type=float, default=1.0, help='stretching function for x')
    parser.add_argument('--M', type=int, default=None, help='number of quantum measurements')
    parser.add_argument('--bin_num_for_naive_histo', type=int, default=32, help='for Monte Carlo integration for eq (27)')
    parser.add_argument('--training_mode', type=str, default='forward_KL', help='loss function for PQC -- default is cross-entropy loss') 
    parser.add_argument('--num_val', type=int, default=10, help='number of validation points')
    parser.add_argument('--num_total_available_data', type=int, default=None, help='number of entire dataset |D|')
    parser.add_argument('--num_tr_data_examples', type=int, default=10, help='number of training data examples')
    parser.add_argument('--M_tr', type=int, default=None, help='number of quantum measurements assumed for training')
    parser.add_argument('--if_pqc_angle_encoding_dnn', dest='if_pqc_angle_encoding_dnn', action='store_true', default=False, help='whether using generalized angle encoding introduced in Fig. 9')    
    parser.add_argument('--pretrained_path', type=str, default=None, help='path for pretrained PQC') 
    parser.add_argument('--bimodal_mode_unsup', type=str, default='strong', help='weak or strong, weakly bimodal and strongly bimodal in the paper')
    parser.add_argument('--used_device', type=str, default='ibmq_quito_with_qem', help='ibmq_qasm_simulator, ibmq_quito_with_qem, ibmq_quito_without_qem')
    parser.add_argument('--pretraining_mode', type=str, default='simulator_pretrained', help='in this project, only considering pretrained PQC on the simulator') 
    parser.add_argument('--qem_mode', type=str, default='with_qem', help='whehter using M3 QEM or not') 
    parser.add_argument('--for_measurements_start_ind_exp', type=int, default='0', help='to continue measurements if it has been stopped in the middle of running') 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('args: ', args)
    args, M_list = automatic_args_setting_unsup(args)
    total_dataset_over_indep_experiments = data_gen_for_unsup_learning(args)
    #### PQC model 
    pqc_model_init = training_pqc_unsup(args, total_dataset_over_indep_experiments)
    given_NC_setting = args.NC_mode[21:]
    for args.M in M_list: 
        print('curr K', args.M)
        args = nearest_neighbor_adaptive_to_M(args, given_NC_setting)
        if args.set_prediction_mode == 'naive_quan':
            quantum_set_predictor = Naive_Predictor(alpha=args.alpha, pqc_model=pqc_model_init, M=args.M, bin_num_for_naive_histo=args.bin_num_for_naive_histo, exp_mode = args.exp_mode)
        elif args.set_prediction_mode == 'CP':
            quantum_set_predictor = QCP(alpha=args.alpha, pqc_model=pqc_model_init, M=args.M, NC_mode=args.NC_mode)  # currently single_c
        else:
            raise NotImplementedError
        dict_for_eval = {}
        dict_for_eval['coverage'] = []
        dict_for_eval['ineff'] = []
        for ind_indep_exp in range(args.num_indep_experiments):
            quantum_set_predictor = bring_saved_measurements(args, quantum_set_predictor, ind_indep_exp) # only if we use fixed measurements!
            curr_entire_dataset = total_dataset_over_indep_experiments['ind_indep_exp_' + str(ind_indep_exp)] # same data set over different quantum to focus on quantum effect!
            if 'naive' in args.set_prediction_mode:
                dict_set_pred_test_inputs = quantum_set_predictor.set_prediction( val_dataset=None, test_inputs = curr_entire_dataset[0]['te'][0])
            elif args.set_prediction_mode == 'CP':
                dict_set_pred_test_inputs = quantum_set_predictor.set_prediction( val_dataset=curr_entire_dataset[0]['val'], test_inputs = curr_entire_dataset[0]['te'][0])
            else:
                raise NotImplementedError
            coverage, ineff = cov_ineff_compute(dict_set_pred_test_inputs=dict_set_pred_test_inputs, test_dataset =curr_entire_dataset[0]['te'])
            dict_for_eval['coverage'].append(coverage)
            dict_for_eval['ineff'].append(ineff.numpy())
        print('avg cov: ', sum(dict_for_eval['coverage'])/len(dict_for_eval['coverage']), 'avg ineff: ', sum(dict_for_eval['ineff'])/len(dict_for_eval['ineff']) )



