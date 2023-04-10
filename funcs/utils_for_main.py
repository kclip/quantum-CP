import os
import torch
import numpy as np
from data_gen.bimodal_gen_with_input import Bimodal_Generator_With_Input
from data_gen.bimodal_gen_no_input import Bimodal_Generator_No_Input



def data_gen_for_unsup_learning(args):
    args.std=0.1
    PATH_FOR_SAVED_DATASET = args.common_dir_for_saved + 'saved_dataset/toy_without_input/' + args.bimodal_mode_unsup + '_bimodal_without_input_num_te_' + str(args.num_te_data_examples) + 'num_indep_exp_' + str(args.num_indep_experiments) + 'num_val_' + str(args.num_val) + 'mean_' + str(args.y_mean_list[1])
    if os.path.isfile(PATH_FOR_SAVED_DATASET):
        print('load saved dataset')
        total_dataset_over_indep_experiments = torch.load(PATH_FOR_SAVED_DATASET)
    else:        
        print('generate dataset')
        toy_dataGen_without_input = Bimodal_Generator_No_Input(dummy_x_input=0, y_mean_list=args.y_mean_list, std=args.std)
        total_dataset_over_indep_experiments = toy_dataGen_without_input.get_entire_data_set(args.num_indep_experiments, args.num_tr_data_examples, args.num_val, args.num_te_data_examples)
        torch.save(total_dataset_over_indep_experiments, PATH_FOR_SAVED_DATASET)
    return total_dataset_over_indep_experiments

def data_gen_for_sup_learning(args):
    args.x_min_max = [-10*args.x_scale, 10*args.x_scale]
    args.std=0.05
    if args.num_te_data_examples == 1:
        PATH_FOR_SAVED_DATASET = args.common_dir_for_saved + 'saved_dataset/toy_with_input/num_te_' + str(args.num_te_data_examples) + 'num_indep_exp_' + str(args.num_indep_experiments) + 'std_' + str(args.std) + 'num_tr_' + str(19990) + 'x_scale_' + str(args.x_scale) + 'num_val_' + str(args.num_val)
        #PATH_FOR_SAVED_DATASET = '/scratch/users/k2142437/QCP/journal_submission/saved_dataset/toy_with_input/num_te_1num_indep_exp_1000std_0.05num_tr_19990x_scale_1.0num_val_10y_rep_tr1'
    else:
        # for vis!!!
        assert args.num_te_data_examples == 2000 # just for safety!
        PATH_FOR_SAVED_DATASET = args.common_dir_for_saved + 'saved_dataset/toy_with_input/num_te_' + str(args.num_te_data_examples) + 'num_indep_exp_' + str(args.num_indep_experiments) + 'std_' + str(args.std) + 'num_tr_' + str(19900) + 'x_scale_' + str(args.x_scale) + 'num_val_' + str(args.num_val)
    
    print('PATH_FOR_SAVED_DATASET', PATH_FOR_SAVED_DATASET)
    if os.path.isfile(PATH_FOR_SAVED_DATASET):
        print('load saved dataset')
        total_dataset_over_indep_experiments = torch.load(PATH_FOR_SAVED_DATASET)
    else:   
        #raise NotImplementedError # this is okay but we already made one data set and we can use this always to avoid unwanted randomness!
        print('generate dataset')
        toy_dataGen_with_input = Bimodal_Generator_With_Input(x_scale = args.x_scale, std=args.std, x_min_max=args.x_min_max) 
        total_dataset_over_indep_experiments = toy_dataGen_with_input.get_entire_data_set(args.num_indep_experiments, args.num_tr_data_examples , args.num_val, args.num_te_data_examples)
        torch.save(total_dataset_over_indep_experiments, PATH_FOR_SAVED_DATASET)
    return total_dataset_over_indep_experiments

def nearest_neighbor_adaptive_to_M(args, given_NC_setting):
    if 'distribution_aware' in args.NC_mode:
        if args.NC_mode == 'distribution_aware':
            pass
        else:
            if given_NC_setting == 'adaptive':
                K_nn_tmp = int(np.ceil(np.sqrt(args.M)))
            else:
                K_nn_tmp = int(min(int(given_NC_setting), args.M))
            args.NC_mode = 'distribution_aware_K_' + str(K_nn_tmp)
            print('original NC mode: ', 'distribution_aware_K_'+given_NC_setting, 'currently: ', args.NC_mode)
    else:
        pass
    return args