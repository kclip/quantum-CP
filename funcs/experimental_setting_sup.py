import os
import pickle
import torch
import numpy as np
from funcs.basic_utils import reset_random_seed
from data_gen.bimodal_gen_with_input import Bimodal_Generator_With_Input

def automatic_args_setting_sup(args):
    assert args.exp_mode == 'toy_with_input'
    assert args.num_total_available_data is not None
    if args.num_total_available_data < 20:
        args.num_val = args.num_total_available_data // 2
    else:
        args.num_val = 10
    args.num_tr_data_examples = args.num_total_available_data - args.num_val
    if 'distribution_aware' in args.NC_mode:
        args.M_tr = 'infty_for_prob'
    elif args.NC_mode == 'distance_aware':
        args.M_tr = 'infty_for_mean'
    elif args.set_prediction_mode == 'naive_quan':
        args.M_tr = 'infty_for_prob'
    else:
        pass
    print('args: ', args)
    print('entire num tr val: ', args.num_tr_data_examples + args.num_val)
    print('num tr: ', args.num_tr_data_examples, 'num val: ', args.num_val)
    reset_random_seed(args.random_seed)
    args.common_dir_for_saved = '../../../../'
    args.num_te_data_examples = 1 #1   #this is what we are doing actually#1 
    args.num_indep_experiments = 1000 #1000 #1000  #this is what we are doing actually#1000 

    if args.set_prediction_mode == 'CP':
        if args.NC_mode == 'distance_aware':
            if args.if_fixed_measurement_for_quantum_run:
                M_list = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000] #[20, 20000]  #[100]#
            else:
                M_list = ['infty_for_mean'] 
        else:
            M_list = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000] #[20, 20000]  #[100]#
    else:
        M_list = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000] #[100]#


    return args, M_list



def training_pqc_or_bring_saved_measurements(args, quantum_set_predictor, measurements_dict, ind_indep_exp, curr_entire_dataset):
    if args.if_fixed_measurement_for_quantum_run:
        quantum_set_predictor.already_trained = True  # we don't need pqc since we already have corresponding measurements !
        quantum_set_predictor.fix_measurement_with_given_dict_per_input(measurements_dict['ind_indep_exp_' + str(ind_indep_exp)])
    else:
        if args.pretrained_path is None:
            if quantum_set_predictor.already_trained:
                pass
            else:
                actual_tr_dataset = determining_actual_training_set(args, curr_entire_dataset)
                quantum_set_predictor.train(exp_mode=args.exp_mode, M_tr=args.M_tr, training_mode=args.training_mode, tr_dataset=actual_tr_dataset, num_tr_iter=args.num_training_iter, lr=args.lr, angle_encoding_mode_for_with_input=args.angle_encoding_mode_for_with_input)            
                quantum_set_predictor.already_trained = True 
                print('training done !!')
        else:
            quantum_set_predictor.pqc_model.load_pqc(args.pretrained_path)
    return quantum_set_predictor

def determining_actual_training_set(args, curr_entire_dataset):
    X_tr_tot, Y_tr_tot = curr_entire_dataset[0]['tr']
    X_tr = X_tr_tot[:args.num_tr_data_examples]
    Y_tr = Y_tr_tot[:args.num_tr_data_examples]
    if args.set_prediction_mode == 'CP':
        actual_tr_dataset = [X_tr, Y_tr]
    else:
        X_val_tot, Y_val_tot = curr_entire_dataset[0]['val']
        X_val = X_val_tot[:args.num_val]
        Y_val = Y_val_tot[:args.num_val]
        if X_val is None:
            X_trval = X_tr
            Y_trval = Y_tr
        else:
            X_trval = torch.cat([X_tr, X_val], dim=0)
            Y_trval = torch.cat([Y_tr, Y_val.repeat(1, Y_tr.shape[1])], dim=0)
        actual_tr_dataset = [X_trval, Y_trval]
    print('training start !! with num tr: ', actual_tr_dataset[0].shape[0])
    return actual_tr_dataset

def generate_measurement_dict_from_saved_measurement_path(args):
    entire_measurements_list = []
    for ind_ten_experiments in range(5):
        M_for_save = 20000
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
        curr_M_saved_path = args.common_dir_for_saved + 'quantum_measurements/toy_with_input/'+ args.angle_encoding_mode_for_with_input + '/' +  set_prediction_mode_for_saved_measurements + '/simulator_pretrained/' + args.used_device + '/'  + 'M_' + str(M_for_save) + '_ind_exp_chunk_of_ten_' + str(ind_ten_experiments) 
       
        if not os.path.isfile(curr_M_saved_path):
            print('fixed measurement case (args.if_fixed_measurement_for_quantum_run=True) can only be done with saved measurement file')
            print('please run /running_on_ibm_quantum/regression_learning_measurements_generation_ibm_quantum.py first')
            raise NotImplementedError
        else:
            pass

       
        with open(curr_M_saved_path,  'rb') as f: 
            curr_fixed_measurement = pickle.load(f)
            entire_measurements_list += curr_fixed_measurement

    measurements_dict = {}
    for ind_indep_exp in range(args.num_indep_experiments):
        measurements_dict['ind_indep_exp_' + str(ind_indep_exp)] = {}
        val_start_ind = ind_indep_exp*(args.num_val+args.num_te_data_examples)
        measurements_dict['ind_indep_exp_' + str(ind_indep_exp)]['val'] = entire_measurements_list[val_start_ind:val_start_ind+args.num_val]
        te_start_ind = ind_indep_exp*(args.num_val+args.num_te_data_examples)+args.num_val
        measurements_dict['ind_indep_exp_' + str(ind_indep_exp)]['te'] = entire_measurements_list[te_start_ind:te_start_ind+args.num_te_data_examples]
    
    return measurements_dict


def custom_split_given_dataset(args, num_te_data_examples, num_indep_experiments, given_total_dataset_over_indep_experiments):

    # we now force it to be 
    args.num_te_data_examples = num_te_data_examples #1# this is for vis: 1000 
    args.num_indep_experiments = num_indep_experiments  #500 #this is for vis: 5 

    val_te_x_whole = []
    val_te_y_whole = []
    #### transform data set into long list
    for ind_indep_exp in range(args.num_indep_experiments):
        curr_entire_dataset = given_total_dataset_over_indep_experiments['ind_indep_exp_' + str(ind_indep_exp)] # same data set over different quantum to focus on quantum effect!
        data_dict = curr_entire_dataset[0]
        val_dataset=data_dict['val']
        te_dataset =data_dict['te']
        val_inputs = val_dataset[0] #(num_val, dim_x)
        val_outputs = val_dataset[1]
        te_inputs = te_dataset[0]
        te_outputs = te_dataset[1]            
        val_te_inputs = torch.cat([val_inputs,te_inputs], dim=0)
        val_te_outputs = torch.cat([val_outputs,te_outputs], dim=0)
        val_te_x_whole.append(val_te_inputs)
        val_te_y_whole.append(val_te_outputs)
    val_te_x_whole_tensor = torch.cat(val_te_x_whole, dim=0)
    val_te_y_whole_tensor = torch.cat(val_te_y_whole, dim=0)

    toy_dataGen_with_input = Bimodal_Generator_With_Input(x_scale = args.x_scale, std=args.std, x_min_max=args.x_min_max) 
    total_dataset_over_indep_experiments = toy_dataGen_with_input.get_entire_data_set(args.num_indep_experiments, 1 , args.num_val, args.num_te_data_examples)
    for ind_indep_exp in range(args.num_indep_experiments):
        val_start_ind = ind_indep_exp*(args.num_val+args.num_te_data_examples)
        X_val_curr = val_te_x_whole_tensor[val_start_ind:val_start_ind+args.num_val]
        Y_val_curr = val_te_y_whole_tensor[val_start_ind:val_start_ind+args.num_val]
        te_start_ind = ind_indep_exp*(args.num_val+args.num_te_data_examples)+args.num_val
        X_te_curr = val_te_x_whole_tensor[te_start_ind:te_start_ind+args.num_te_data_examples]
        Y_te_curr = val_te_y_whole_tensor[te_start_ind:te_start_ind+args.num_te_data_examples]
        total_dataset_over_indep_experiments['ind_indep_exp_' + str(ind_indep_exp)][0]['val'] = [X_val_curr, Y_val_curr]
        total_dataset_over_indep_experiments['ind_indep_exp_' + str(ind_indep_exp)][0]['te'] = [X_te_curr, Y_te_curr]
    return args, total_dataset_over_indep_experiments
