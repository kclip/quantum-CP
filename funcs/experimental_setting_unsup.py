import os
import pickle
import torch
import numpy as np
from funcs.basic_utils import reset_random_seed
from quantum_circuit.PQC import PQC_multi_qbit
from set_predictors.quantum_conformal_prediction import QCP

def automatic_args_setting_unsup(args):
    assert args.exp_mode == 'toy_without_input'
    if 'distribution_aware' in args.NC_mode:
        args.M_tr = 'infty_for_prob'
    elif args.NC_mode == 'distance_aware':
        args.M_tr = 'infty_for_mean'
    elif args.set_prediction_mode == 'naive_quan':
        args.M_tr = 'infty_for_prob'
    else:
        pass
    args.common_dir_for_saved = '../../../../'
    reset_random_seed(args.random_seed)
    args.num_te_data_examples = 1 # we take expectation over te by computing the true analytical integration
    args.num_indep_experiments = 1000

    if args.bimodal_mode_unsup == 'strong':
        args.y_mean_list = [-0.75, 0.75] ## strong bimodal
    elif args.bimodal_mode_unsup == 'weak':
        args.y_mean_list = [-0.15, 0.15]
    else:
        raise NotImplementedError
    if args.if_fixed_measurement_for_quantum_run:
        args.if_training_toy_without_input = False
    else:
        if args.pretrained_path is None:
            args.if_training_toy_without_input = True
        else:
            args.if_training_toy_without_input = False
    if args.set_prediction_mode == 'CP':
        if args.NC_mode == 'distance_aware': # this is DCP; DCP assumes infinite shots
            M_list = ['infty_for_mean']
            #M_list = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000] # for ibmq!
        else:
            M_list = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000]
    else:
        M_list = [0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000]
    return args, M_list

def training_pqc_unsup(args, total_dataset_over_indep_experiments):
    if args.if_training_toy_without_input:
        pqc_model_init = PQC_multi_qbit(num_layers=args.num_layers, num_qubits=5, observation_min_max=[-1,1], if_pqc_angle_encoding_dnn=False, if_vanilla_angle_encoding=True)
        # just for training load QCP class
        quantum_set_predictor_for_training = QCP(alpha=args.alpha, pqc_model=pqc_model_init, M=args.M, NC_mode=None, exp_mode = args.exp_mode)
        ####training 
        curr_entire_dataset_for_tr = total_dataset_over_indep_experiments['ind_indep_exp_' + str(0)] # train once since any pre-trained model can be applied
        if args.set_prediction_mode == 'CP':
            actual_tr_dataset = curr_entire_dataset_for_tr[0]['tr']
        else:
            X_tr, Y_tr = curr_entire_dataset_for_tr[0]['tr']
            X_val, Y_val = curr_entire_dataset_for_tr[0]['val']
            X_trval = torch.cat([X_tr, X_val], dim=0)
            Y_trval = torch.cat([Y_tr, Y_val.repeat(1, Y_tr.shape[1])], dim=0)
            actual_tr_dataset = [X_trval, Y_trval]
        quantum_set_predictor_for_training.train(exp_mode=args.exp_mode, M_tr=args.M_tr, training_mode=args.training_mode, tr_dataset=actual_tr_dataset, num_tr_iter=args.num_training_iter, lr=args.lr, angle_encoding_mode_for_with_input=None, bimodal_mode_unsup = args.bimodal_mode_unsup)            
        print('training done !!')
        pqc_model_init = quantum_set_predictor_for_training.pqc_model
    else:
        if args.if_fixed_measurement_for_quantum_run:
            pqc_model_init = None # we don't need it since we already have all the measurements!
        else:
            pqc_model_init.load_pqc(args.pretrained_path)
    return pqc_model_init


def bring_saved_measurements(args, quantum_set_predictor, ind_indep_exp):
    if args.if_fixed_measurement_for_quantum_run:
        if args.M == 0:
            curr_fixed_measurement = [[]]*args.num_indep_experiments # empty sets
        else:
            if args.set_prediction_mode == 'CP':
                if args.NC_mode == 'distance_aware':
                    if args.M == 'infty_for_mean':
                        raise NotImplementedError # cannot use saved finite measurements for infinite case!
                    else:
                        set_prediction_mode_for_saved_measurements = 'DCP'
                else:
                    set_prediction_mode_for_saved_measurements = 'CP'
            else:
                set_prediction_mode_for_saved_measurements = args.set_prediction_mode
            
            saved_measurements_fixed_input_path =  args.common_dir_for_saved + 'quantum_measurements/toy_without_input/mean_' + str(args.y_mean_list[1]) + '/' + set_prediction_mode_for_saved_measurements + '/' + args.pretraining_mode + '/' + args.used_device + '/'
            
            if not os.path.isfile(saved_measurements_fixed_input_path + 'M_' + str(args.M)):
                print('fixed measurement case (args.if_fixed_measurement_for_quantum_run=True) can only be done with saved measurement file')
                print('please run /running_on_ibm_quantum/density_learning_measurements_generation_ibm_quantum.py first')
                raise NotImplementedError
            else:
                pass
            with open(saved_measurements_fixed_input_path + 'M_' + str(args.M),  'rb') as f: 
                curr_fixed_measurement = pickle.load(f)
        quantum_set_predictor.fix_measurement_with_given_list(curr_fixed_measurement[ind_indep_exp])
    else:
        pass # we already 
    return quantum_set_predictor