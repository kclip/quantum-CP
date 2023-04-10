import torch
import argparse
import numpy as np
#from quantum_circuit.PQC import PQC_multi_qbit
from funcs.utils_for_eval import cov_ineff_compute
from set_predictors.quantum_conformal_prediction import QCP
from set_predictors.quantum_naive_set_prediction_using_quantile import Naive_Predictor
from funcs.utils_for_main import data_gen_for_sup_learning, nearest_neighbor_adaptive_to_M
from funcs.experimental_setting_sup import automatic_args_setting_sup, custom_split_given_dataset, generate_measurement_dict_from_saved_measurement_path, determining_actual_training_set, training_pqc_or_bring_saved_measurements
## for IBMQ
from qiskit_ibm_runtime import Sampler, Session, Options
from quantum_circuit.PQC_with_qiskit import PQC_multi_qbit_qiskit

def parse_args():
    parser = argparse.ArgumentParser(description='QCP')
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
    parser.add_argument('--qem_mode', type=str, default='with_qem', help='whehter using M3 QEM or not') 
    parser.add_argument('--for_measurements_start_ind_exp', type=int, default='0', help='to continue measurements if it has been stopped in the middle of running') 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args, val_te_whole_tensor, pqc_model, resil_level, saved_measurements_fixed_input_dir = pre_setting(args)
    num_val_te = num_val_data_examples+num_te_data_examples 
    one_chunck_size = num_val_te*100  # 1100 examples ok!
    M_for_save = 20000
    with Session(service=pqc_model.service, backend=args.used_device): 
        print(args.qem_mode)
        options = Options(resilience_level=resil_level)
        for ind_ten_experiments in range(args.for_measurements_start_ind_exp, 10):
            print('ind_ten_experiments: ', ind_ten_experiments)
            print('M (shots) for save: ', M_for_save)
            curr_M_saved_path = saved_measurements_fixed_input_dir + 'M_' + str(M_for_save) + '_ind_exp_chunk_of_ten_' + str(ind_ten_experiments) 
            sampler = Sampler(options=options)
            x_batch = val_te_whole_tensor[ind_ten_experiments*one_chunck_size : ind_ten_experiments*one_chunck_size +  one_chunck_size]
            t0 = time.time()
            measurement_list_batch = pqc_model.M_shots_measurement(x_batch, M_for_save, sampler)
            t1 = time.time()
            print('time spent during measurement:',  t1-t0)
            with open(curr_M_saved_path, 'wb') as f: 
                pickle.dump(measurement_list_batch, f)



def concat_val_te(num_indep_experiments, total_dataset_over_indep_experiments):
    val_te_whole = []
    #### transform data set into long list
    for ind_indep_exp in range(num_indep_experiments):
        curr_entire_dataset = total_dataset_over_indep_experiments['ind_indep_exp_' + str(ind_indep_exp)] # same data set over different quantum to focus on quantum effect!
        data_dict = curr_entire_dataset[0]
        val_dataset=data_dict['val']
        test_inputs = data_dict['te'][0]
        val_inputs = val_dataset[0] #(num_val, dim_x)
        te_inputs = test_inputs #(num_te, dim_x)
        val_te_inputs = torch.cat([val_inputs,te_inputs], dim=0)
        val_te_whole.append(val_te_inputs)
    val_te_whole_tensor = torch.cat(val_te_whole, dim=0)
    return val_te_whole_tensor
    
def pre_setting(args):
    common_dir_for_saved = '../../../../../'
    args, M_list = automatic_args_setting_sup(args)
    total_dataset_over_indep_experiments = data_gen_for_sup_learning(args)
    val_te_whole_tensor = concat_val_te(num_indep_experiments, total_dataset_over_indep_experiments)
    pqc_model = PQC_multi_qbit_qiskit(num_layers=args.num_layers, num_qubits=5, observation_min_max=[-1,1], if_pqc_angle_encoding_dnn=args.if_pqc_angle_encoding_dnn, if_vanilla_angle_encoding=None, angle_encoding_mode_for_with_input=args.angle_encoding_mode_for_with_input)
    ###############
    if args.pretrained_path is not None:
        pqc_model.load_pqc(args.pretrained_path)
    else:
        print('in this project, we consider pre-trained PQC only')
        raise NotImplementedError
    
    if args.qem_mode == 'with_qem':
        resil_level = 1
    elif args.qem_mode == 'without_qem':
        resil_level = 0
    else:
        raise NotImplementedError
    
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

    saved_measurements_fixed_input_dir =  common_dir_for_saved + 'quantum_measurements/toy_with_input/'+ args.angle_encoding_mode_for_with_input + '/' +  set_prediction_mode_for_saved_measurements + '/simulator_pretrained/' + args.used_device + '_' + args.qem_mode  + '/'  + 'M_' + str(M_for_save) + '_ind_exp_chunk_of_ten_' + str(ind_ten_experiments) 
    
    if os.path.isdir(saved_measurements_fixed_input_dir):
        pass
    else:
        os.makedirs(saved_measurements_fixed_input_dir)

    return args, val_te_whole_tensor, pqc_model, resil_level, saved_measurements_fixed_input_dir