import torch
import argparse
import numpy as np
from quantum_circuit.PQC import PQC_multi_qbit
from funcs.utils_for_eval import cov_ineff_compute
from set_predictors.quantum_conformal_prediction import QCP
from set_predictors.quantum_naive_set_prediction_using_quantile import Naive_Predictor
from funcs.utils_for_main import data_gen_for_sup_learning, nearest_neighbor_adaptive_to_M
from funcs.experimental_setting_sup import automatic_args_setting_sup, custom_split_given_dataset, generate_measurement_dict_from_saved_measurement_path, determining_actual_training_set, training_pqc_or_bring_saved_measurements

def parse_args():
    parser = argparse.ArgumentParser(description='QCP for supervised learning')
    parser.add_argument('--alpha', type=float, default=0.1, help='predetermined miscoverage level')
    parser.add_argument('--set_prediction_mode', type=str, default='CP', help='either naive_quan (for naive set predictor),  or CP')
    parser.add_argument('--NC_mode', type=str, default='distribution_aware_K_adaptive', help='scoring function, default is the proposed scoring function for QCP') 
    parser.add_argument('--random_seed', type=int, default=10, help='random seed')
    parser.add_argument('--exp_mode', type=str, default='toy_with_input', help='toy_without_input, toy_with_input') 
    parser.add_argument('--num_training_iter', type=int, default=10000, help='training iteration')  
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate') 
    parser.add_argument('--num_layers', type=int, default=5, help='number of layers (L) for PQC')
    parser.add_argument('--if_fixed_measurement_for_quantum_run', dest='if_fixed_measurement_for_quantum_run', action='store_true', default=False, help='using fixed measurements from saved file')    
    parser.add_argument('--x_scale', type=float, default=1.0, help='stretching function for x')
    parser.add_argument('--M', type=int, default=None, help='number of quantum measurements')
    parser.add_argument('--bin_num_for_naive_histo', type=int, default=32, help='for Monte Carlo integration for eq (34)')
    parser.add_argument('--training_mode', type=str, default='forward_KL', help='loss function for PQC -- default is cross-entropy loss') 
    parser.add_argument('--num_val', type=int, default=10, help='number of validation points')
    parser.add_argument('--num_total_available_data', type=int, default=None, help='number of entire dataset |D|')
    parser.add_argument('--num_tr_data_examples', type=int, default=10, help='number of training data examples')
    parser.add_argument('--M_tr', type=int, default=None, help='number of quantum measurements assumed for training')
    parser.add_argument('--if_pqc_angle_encoding_dnn', dest='if_pqc_angle_encoding_dnn', action='store_true', default=False, help='whether using generalized angle encoding introduced in Fig. 9')    
    parser.add_argument('--pretrained_path', type=str, default=None, help='path for pretrained PQC') 
    parser.add_argument('--angle_encoding_mode_for_with_input', type=str, default='learned_nonlinear', help='learned_nonlinear, learned_linear, fixed') 
    parser.add_argument('--used_device', type=str, default='ibmq_quito_with_qem', help='ibmq_qasm_simulator, ibmq_quito_with_qem, ibmq_quito_without_qem')
    parser.add_argument('--pretraining_mode', type=str, default='simulator_pretrained', help='in this project, only considering pretrained PQC on the simulator') 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args, M_list = automatic_args_setting_sup(args)
    total_dataset_over_indep_experiments = data_gen_for_sup_learning(args)
    if args.if_fixed_measurement_for_quantum_run:
        # this is useful when using same data set but want different split
        desired_num_te_data_examples = 1
        desired_num_indep_experiments = 500
        args, total_dataset_over_indep_experiments = custom_split_given_dataset(args, desired_num_te_data_examples, desired_num_indep_experiments, total_dataset_over_indep_experiments)
        measurements_dict = generate_measurement_dict_from_saved_measurement_path(args)
    else:
        measurements_dict = None
    #### PQC model 
    pqc_model_init = PQC_multi_qbit(num_layers=args.num_layers, num_qubits=5, observation_min_max=[-1,1], if_pqc_angle_encoding_dnn=args.if_pqc_angle_encoding_dnn, if_vanilla_angle_encoding=None, angle_encoding_mode_for_with_input=args.angle_encoding_mode_for_with_input)
    if args.set_prediction_mode == 'naive_quan':
        quantum_set_predictor = Naive_Predictor(alpha=args.alpha, pqc_model=pqc_model_init, M=args.M, bin_num_for_naive_histo=args.bin_num_for_naive_histo, exp_mode = args.exp_mode)
    elif args.set_prediction_mode == 'CP':
        quantum_set_predictor = QCP(alpha=args.alpha, pqc_model=pqc_model_init, M=args.M, NC_mode=args.NC_mode)
    else:
        raise NotImplementedError

    given_NC_setting = args.NC_mode[21:]
    for args.M in M_list:
        print('K:', args.M)
        args = nearest_neighbor_adaptive_to_M(args, given_NC_setting)
        quantum_set_predictor.M = args.M  # training does not depend on args.M -- this only matters for testing 
        quantum_set_predictor.NC_mode = args.NC_mode
        dict_for_eval = {}
        dict_for_eval['coverage'] = []
        dict_for_eval['ineff'] = []
        for ind_indep_exp in range(args.num_indep_experiments):
            curr_entire_dataset = total_dataset_over_indep_experiments['ind_indep_exp_' + str(ind_indep_exp)] # same data set over different quantum to focus on quantum effect!
            # training or load pre-trained model
            quantum_set_predictor = training_pqc_or_bring_saved_measurements(args, quantum_set_predictor, measurements_dict, ind_indep_exp, curr_entire_dataset)
            # computing error bars for test data with validation set
            X_val_tot_eval, Y_val_tot_eval = curr_entire_dataset[0]['val']
            actual_val_dataset = [X_val_tot_eval[:args.num_val], Y_val_tot_eval[:args.num_val]]
            if 'naive' in args.set_prediction_mode:
                dict_set_pred_test_inputs = quantum_set_predictor.set_prediction( val_dataset=None, test_inputs = curr_entire_dataset[0]['te'][0])
            elif args.set_prediction_mode == 'CP':
                dict_set_pred_test_inputs = quantum_set_predictor.set_prediction( val_dataset=actual_val_dataset, test_inputs = curr_entire_dataset[0]['te'][0])
            else:
                raise NotImplementedError
            coverage, ineff = cov_ineff_compute(dict_set_pred_test_inputs=dict_set_pred_test_inputs, test_dataset =curr_entire_dataset[0]['te'])
            dict_for_eval['coverage'].append(coverage)
            dict_for_eval['ineff'].append(ineff.numpy())
        print('avg cov: ', sum(dict_for_eval['coverage'])/len(dict_for_eval['coverage']), 'avg ineff: ', sum(dict_for_eval['ineff'])/len(dict_for_eval['ineff']) )

