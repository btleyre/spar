from data_generate import load_data
from utils import set_seed, stats_values, get_unique_file_name, write_result, write_model
from config import dataset_defaults

from tokenize import Special
import algorithm
from models import Learner, Learner_TimeSeries,Learner_Dti_dg, Learner_RCF_MNIST, RepresentationLearner, Learner_TMLR
from torchvision import models
import copy
import numpy as np
import torch
import scipy
from scipy.stats import chi2
import argparse
import random

import pickle
import os
import matplotlib.pyplot as plt
import time
import io
import torch.nn as nn
import pandas as pd


############ cmd process ##############
parser = argparse.ArgumentParser(description='kde + mixup')
parser.add_argument('--result_root_path', type = str, default="../../result/",
                    help="path to store the results")
parser.add_argument('--result_csv_path', type = str, default="../result_csvs/",
                    help="path to store the results")
parser.add_argument('--dataset', type=str, default='NO2', 
                    help='dataset')
parser.add_argument('--mixtype', type=str, default='random',
                    help="random or kde or erm")
parser.add_argument('--use_manifold', type=int, default=1,
                    help='use manifold mixup or not')
parser.add_argument('--seed', type=int, default=0,
                    help="seed")
parser.add_argument('--gpu', type=int, default=0,
                    help="train on which cuda device")
parser.add_argument('--spar_alpha', type=float, default=0.99,
                    help="The risk parameter for SpAR.")
parser.add_argument('--search_lr', type=float, default=None,
                    help="The new learning rate we are trying in hyperparam search")
parser.add_argument('--search_artifact_base_path', type=str, default=None,
                    help="The base path for artifacts from this param setting")
parser.add_argument("--use_bias", default=False, action='store_true')

#### kde parameter ####
parser.add_argument('--kde_bandwidth', type=float, default=1.0,
                    help="bandwidth")
parser.add_argument('--kde_type', type=str, default='gaussian', help = 'gaussian or tophat')
parser.add_argument('--batch_type', default=0, type=int, help='1 for y batch and 2 for x batch and 3 for representation')

#### verbose ####
parser.add_argument('--show_process', type=int, default = 1,
                    help = 'show rmse and r^2 in the process')
parser.add_argument('--show_setting', type=int, default = 1,
                    help = 'show setting')

#### model read & write ####
parser.add_argument('--read_best_model', type=int, default=0, help='read from original model')
parser.add_argument('--store_model', type=int, default=1, 
                    help = 'store model or not')

########## data path, for RCF_MNIST and TimeSeries #########
parser.add_argument('--data_dir', type = str, help = 'for RCF_MNIST and TimeSeries')

parser.add_argument('--ts_name', type=str,  default='',
                    help='ts dataset name')

### Full batch for other regularizers
parser.add_argument("--full_batch", default=False, action='store_true')

### CORAL related arguments
parser.add_argument('--coral_penalty_weight', type=float, default=1.0)
parser.add_argument("--train_coral", default=False, action='store_true')

### DANN related arguments
parser.add_argument('--disc_penalty_weight', type=float, default=1.0)
parser.add_argument('--disc_lr', type=float, default=0.001)
parser.add_argument("--train_dann", default=False, action='store_true')


########## cmd end ############

args = parser.parse_args()
args.cuda = torch.cuda.is_available() # for ts_data init function
args_dict = args.__dict__
dict_name = args.dataset
if args.dataset == 'TimeSeries':
    dict_name += '-' + args.ts_name
args_dict.update(dataset_defaults[dict_name])
# Swap out the learning rate with the search LR if its been specified
if args_dict['search_lr'] is not None:
    args_dict['optimiser_args']['lr'] = args_dict['search_lr']

args = argparse.Namespace(**args_dict)
if args.show_setting: # basic information
    for k in dataset_defaults[dict_name].keys():
        print(f'{k}: {dataset_defaults[dict_name][k]}')

########## device ##########

if torch.cuda.is_available() and args.gpu != -1:
    torch.cuda.set_device('cuda:'+str(args.gpu))
    device = torch.device('cuda:'+str(args.gpu))
    if args.show_setting:
        print(device)
else:
    device = torch.device('cpu')
    if args.show_setting:
        print("use cpu")

set_seed(args.seed) # init set

####### mkdir result path ########
result_root = args.result_root_path
result_csv_root = args.result_csv_path

if not os.path.exists(result_root):
    os.mkdir(result_root)

result_path = result_root + f"{args.dataset}/"
if not os.path.exists(result_path):
    os.mkdir(result_path)

if args.search_artifact_base_path is None:
    result_csv_path = result_csv_root + f"{args.dataset}/{args.kde_bandwidth}/"
    if not os.path.exists(result_csv_path):
        os.mkdir(result_csv_path)


def compute_pseudoinvese_soln(X_train_representations, Y_train_labels):
    """Return the pinv soln, X^+Y."""
    train_pinv_soln = np.matmul(
        scipy.linalg.pinv(X_train_representations),
        Y_train_labels
    ).squeeze() 
    return train_pinv_soln[:, None]


def spar_chi_adaptation(X_train_representations, Z_test_representations, Y_train_labels, sigma_squared):
    """Adapt the OLS regressor according to our SpAR-Chi approach.

    Parameters:
        X_train_representations: torch tensor. The training representations.
        Z_test_representations: torch tensor. The test representations.
        Y_train_labels: torch tensor. The training labels.
        sigma_squared: float. The estimated variance of the label noise.

    Returns: torch tensor. The adapted regressor.

    """
    # Perform SVD to get the right singular vectors. These will be used
    # to construct the subspaces that we're projecting into.
    u_x, s_x, vh_x = np.linalg.svd(X_train_representations, full_matrices=False)
    squared_s_x = s_x**2
    u_z, s_z, vh_z = np.linalg.svd(Z_test_representations, full_matrices=False)
    squared_s_z = s_z**2

    train_rank = np.linalg.matrix_rank(X_train_representations)
    u_x = u_x[:, :train_rank]
    s_x = s_x[:train_rank]
    squared_s_x = squared_s_x[:train_rank]
    print("vh_x is {}".format(vh_x.shape))
    print(train_rank)
    null_vh_x = copy.deepcopy(vh_x)[train_rank:]
    vh_x = vh_x[:train_rank]
    ood_rank = np.linalg.matrix_rank(Z_test_representations)

    # Get the test eigenvector variances
    eig_correlations = np.matmul(vh_x, vh_z.transpose())
    eigenratio_matrix = np.matmul((1/squared_s_x)[:, None], squared_s_z[None, :])
    eigenmetric_matrix = (eig_correlations**2)*eigenratio_matrix
    test_eigvec_variances = np.sum(eigenmetric_matrix, axis=0)*sigma_squared


    # Calculate the pseudoinverse solution, and use it to estimate the bias term
    train_pinv_soln = compute_pseudoinvese_soln(X_train_representations, Y_train_labels).squeeze()
    test_eigvec_biases = (np.matmul(vh_z, train_pinv_soln).squeeze()*s_z)**2

    # Conduct the Chi^2 based comparisons, and use this to edit the regressor.
    chi2_threshold = chi2.ppf(float(args.spar_alpha), 1)*test_eigvec_variances
    chi2_remove_map = test_eigvec_biases.squeeze() <= chi2_threshold.squeeze()
    print(test_eigvec_biases.squeeze())
    print(chi2_threshold.squeeze())
    print(chi2_remove_map)

    num_chi_2_evecs_retained = np.sum((~(chi2_remove_map)).astype(int))

    chi2_eigvecs_to_be_removed = vh_z[chi2_remove_map, :]

    if chi2_eigvecs_to_be_removed.shape[0] == 0:
        # In this case, no evecs to be removed, so we just keep it as ols
        chi2_w_proj = train_pinv_soln
    else:

        if chi2_eigvecs_to_be_removed.shape[0] == 1:
            chi2_bad_eigvec_projection_weights = np.matmul(chi2_eigvecs_to_be_removed, train_pinv_soln)[:, None]
        else:
            chi2_bad_eigvec_projection_weights = np.matmul(chi2_eigvecs_to_be_removed, train_pinv_soln).squeeze()[:, None]

        print("Chi shapes")
        print(chi2_eigvecs_to_be_removed.shape)
        print(chi2_bad_eigvec_projection_weights.shape)
        chi2_remove_vector = np.sum(
            chi2_bad_eigvec_projection_weights*chi2_eigvecs_to_be_removed,
            axis=0
        ).squeeze()
        print(chi2_remove_vector.shape)
        print(train_pinv_soln.shape)
        chi2_w_proj = train_pinv_soln - chi2_remove_vector
        print(chi2_w_proj.shape)

    # Structure the output as a Dx1 vector.
    print(chi2_w_proj.shape)
    assert len(chi2_w_proj.shape) == 1
    return chi2_w_proj[:, None]


def load_model(args, ts_data):
    if args.dataset == 'TimeSeries':
        model = Learner_TimeSeries(args=args,data=ts_data).to(device)
    elif args.dataset == 'Dti_dg':
        model = Learner_Dti_dg(hparams=None).to(device)
    elif args.dataset == 'RCF_MNIST':
        model = Learner_RCF_MNIST(args=args).to(device)
    elif args.dataset == 'ChairAngle_Tails':
        model = Learner_TMLR(args=args).to(device)
    else:
        if args.train_coral or args.train_dann:
            model = RepresentationLearner(args=args).to(device)
        else:
            model = Learner(args=args).to(device)
    
    if args.show_setting:
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of parameters: %d' % nParams)
    return model


def main():
    t1 = time.time()
    best_model_dict = {}
    data_packet, ts_data = load_data(args)
    if args.show_setting:
        print('load dataset success, use time = {:.4f}'.format(time.time() - t1))
        print(f'args.mixtype = {args.mixtype}, Use_manifold = {args.use_manifold}')
    
    set_seed(args.seed) # seed aligned 

    if args.read_best_model == 0: # normal train
        #### model ####
        model = load_model(args,ts_data)
        if args.show_setting:
            print('load untrained model done')
            print(args)
        
        all_begin = time.time()

        #### get mixup sample rate among data ####
        if args.mixtype == 'kde':
            mixup_idx_sample_rate = algorithm.get_mixup_sample_rate(args, data_packet, device)
        else:
            mixup_idx_sample_rate = None
        
        sample_use_time = time.time() - all_begin
        print('sample use time = {:.4f}'.format(sample_use_time))

        #### train model ####
        if args.train_coral:
            best_model_dict['rmse'], best_model_dict['r'] = algorithm.train_deep_coral(args, model, data_packet, mixup_idx_sample_rate, ts_data, device,
                                                            coral_penalty_weight=args.coral_penalty_weight,
                                                            full_batch=args.full_batch)
        elif args.train_dann:
            best_model_dict['rmse'], best_model_dict['r'] = algorithm.train_dann(args, model, data_packet, mixup_idx_sample_rate, ts_data, device,
                                                            disc_penalty_weight=args.disc_penalty_weight,
                                                            full_batch=args.full_batch)                     
        else:
            best_model_dict['rmse'], best_model_dict['r'] = algorithm.train(args, model, data_packet, args.mixtype != "erm", mixup_idx_sample_rate,ts_data, device)
        print('='*30 + ' single experiment result ' + '=' * 30)
        result_dict_best = algorithm.test(args, best_model_dict[args.metrics], data_packet['x_test'], data_packet['y_test'],
                                        'seed = ' + str(args.seed) + ': Final test for best ' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                                        args.show_process, all_begin, device)
        # include the validation performance
        result_dict_best_val_dict = algorithm.test(args, best_model_dict[args.metrics], data_packet['x_valid'], data_packet['y_valid'],
                                        'seed = ' + str(args.seed) + ': Final val for best ' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                                        args.show_process, all_begin, device)
        result_dict_best['val_rmse'] = result_dict_best_val_dict['rmse']
        result_dict_best['val_r'] = result_dict_best_val_dict['r']
        # Encode some representations for adaptation later
        x_train_reps, y_train_labels = algorithm.save_reps_and_labels(args, best_model_dict[args.metrics], data_packet['x_train'], data_packet['y_train'], device)
        x_valid_reps, y_valid_labels = algorithm.save_reps_and_labels(args, best_model_dict[args.metrics], data_packet['x_valid'], data_packet['y_valid'], device)
        z_test_reps, y_test_labels = algorithm.save_reps_and_labels(args, best_model_dict[args.metrics], data_packet['x_test'], data_packet['y_test'], device)

        # Estimate the variance of the noise using the pinv soln
        train_pinv_soln = compute_pseudoinvese_soln(x_train_reps, y_train_labels)
        algorithm.test(args, model, data_packet['x_train'], data_packet['y_train'],
                        'seed = ' + str(args.seed) + ': Training model on train' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                        args.show_process, all_begin, device)
        algorithm.test(args, model, data_packet['x_valid'], data_packet['y_valid'],
                        'seed = ' + str(args.seed) + ': Training model on val' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                        args.show_process, all_begin, device)

        
        if args.dataset in ['CommunitiesAndCrime', 'SkillCraft']:
            new_model = torch.nn.Linear(128, 1, bias=False)
            new_model = new_model.to(device)
            new_state_dict = {'weight': best_model_dict[args.metrics].fclayer[0].weight}
        elif args.dataset in ['RCF_MNIST', 'ChairAngle_Tails']:
            new_model = torch.nn.Linear(512, 1, bias=False)
            new_model = new_model.to(device)
            new_state_dict = {'weight': best_model_dict[args.metrics].fc.weight}
        new_model.load_state_dict(new_state_dict)
        new_model = new_model.to(device)

        print("Test linear original model")
        algorithm.linear_model_test(args, new_model, x_valid_reps, y_valid_labels,
                        'seed = ' + str(args.seed) + ': Linear evaluation: train model on val' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                        args.show_process, all_begin, device)
        

        if args.dataset in ['CommunitiesAndCrime', 'SkillCraft']:
            ols_model = torch.nn.Linear(128, 1, bias=False)
        elif args.dataset in ['RCF_MNIST', 'ChairAngle_Tails']:
            ols_model = torch.nn.Linear(512, 1, bias=False)
        #ols_model = torch.nn.Linear(128, 1, bias=False)
        ols_model = ols_model.to(device)
        new_state_dict = {'weight': torch.from_numpy(train_pinv_soln).transpose(1, 0)}
        ols_model.load_state_dict(new_state_dict)
        ols_model = ols_model.to(device)

        ols_train_perf_dict = algorithm.linear_model_test(args, ols_model, x_train_reps, y_train_labels,
                        'seed = ' + str(args.seed) + ': OLS evaluation: OLS model on train reps and labels' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                        args.show_process, all_begin, device)
        print("Train reps shape: {}".format(x_train_reps.shape))
        sigma_squared_estimate = ols_train_perf_dict['total_se']/x_train_reps.shape[0]
        print("The estimate for sigma^2 is: {}".format(sigma_squared_estimate))
        # Create a projected regressor
        w_proj = spar_chi_adaptation(x_train_reps, z_test_reps, y_train_labels, sigma_squared=sigma_squared_estimate)

        if args.dataset in ['CommunitiesAndCrime', 'SkillCraft']:
            w_proj_model = torch.nn.Linear(128, 1, bias=False)
        elif args.dataset in ['RCF_MNIST', 'ChairAngle_Tails']:
            w_proj_model = torch.nn.Linear(512, 1, bias=False)
        #w_proj_model = torch.nn.Linear(128, 1, bias=False)
        w_proj_model = w_proj_model.to(device)
        new_state_dict = {'weight': torch.from_numpy(w_proj).transpose(1, 0)}
        w_proj_model.load_state_dict(new_state_dict)
        w_proj_model = w_proj_model.to(device)

        ols_test_perf_dict = algorithm.linear_model_test(args, ols_model, z_test_reps, y_test_labels,
                        'seed = ' + str(args.seed) + ': OLS on test reps and labels' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                        args.show_process, all_begin, device)

        w_proj_perf_dict = algorithm.linear_model_test(args, w_proj_model, z_test_reps, y_test_labels,
                        'seed = ' + str(args.seed) + ': W proj on test reps and labels' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                        args.show_process, all_begin, device)

        # add the validation performance

        ols_test_perf_dict_val = algorithm.linear_model_test(args, ols_model, x_valid_reps, y_valid_labels,
                        'seed = ' + str(args.seed) + ': OLS on val reps and labels' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                        args.show_process, all_begin, device)

        ols_test_perf_dict['val_rmse'] = ols_test_perf_dict_val['rmse']
        ols_test_perf_dict['val_r'] = ols_test_perf_dict_val['r']

        w_proj_perf_dict_val = algorithm.linear_model_test(args, w_proj_model, x_valid_reps, y_valid_labels,
                        'seed = ' + str(args.seed) + ': W proj on val reps and labels' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                        args.show_process, all_begin, device)

        w_proj_perf_dict['val_rmse'] = w_proj_perf_dict_val['rmse']
        w_proj_perf_dict['val_r'] = w_proj_perf_dict_val['r']

        if args.dataset != 'ChairAngle_Tails':
            print("SGD Worst group")
            algorithm.cal_worst_acc(args,data_packet,best_model_dict[args.metrics], result_dict_best, all_begin,ts_data,device)
            print("Linear test SGD Worst group")
            algorithm.linear_cal_worst_acc(args,data_packet,best_model_dict[args.metrics], copy.deepcopy(result_dict_best), all_begin,ts_data,device, new_model)
            print("Worst group w_proj model")
            algorithm.linear_cal_worst_acc(args,data_packet,best_model_dict[args.metrics], w_proj_perf_dict, all_begin,ts_data,device, w_proj_model)
            print("Worst group ols model")
            algorithm.linear_cal_worst_acc(args,data_packet,best_model_dict[args.metrics], ols_test_perf_dict, all_begin,ts_data,device, ols_model)

        result_dict_best_frame_version = copy.deepcopy(result_dict_best)


        # Save as csvs, start by adding an entry indicating the seed and method
        result_dict_best_frame_version['seed'] = args.seed
        result_dict_best_frame_version['method'] = args.mixtype
        result_dict_best_frame_version['lr'] = args.optimiser_args['lr']
        result_dict_best_frame_version['bw'] = args.kde_bandwidth
        w_proj_perf_dict['seed'] = args.seed
        w_proj_perf_dict['method'] = "{}_spar_{}".format(args.mixtype, args.spar_alpha)
        w_proj_perf_dict['lr'] = args.optimiser_args['lr']
        w_proj_perf_dict['bw'] = args.kde_bandwidth
        w_proj_perf_dict['spar_alpha'] = args.spar_alpha        
        ols_test_perf_dict['seed'] = args.seed
        ols_test_perf_dict['method'] = "{}_ols".format(args.mixtype)
        ols_test_perf_dict['lr'] = args.optimiser_args['lr']
        ols_test_perf_dict['bw'] = args.kde_bandwidth
        if args.train_dann:
            result_dict_best_frame_version['penalty_weight'] = args.disc_penalty_weight
            w_proj_perf_dict['penalty_weight'] = args.disc_penalty_weight
            ols_test_perf_dict['penalty_weight'] = args.disc_penalty_weight
        elif args.train_coral:
            result_dict_best_frame_version['penalty_weight'] = args.coral_penalty_weight
            w_proj_perf_dict['penalty_weight'] = args.coral_penalty_weight
            ols_test_perf_dict['penalty_weight'] = args.coral_penalty_weight

        artifact_path = args.search_artifact_base_path if args.search_artifact_base_path is not None else result_csv_path
        print("{}seed_{}_{}_spar_{}.csv".format(artifact_path, args.seed, args.mixtype, args.spar_alpha))
        pd.DataFrame(result_dict_best_frame_version, index=[0]).to_csv("{}seed_{}_{}.csv".format(artifact_path, args.seed, args.mixtype))
        pd.DataFrame(w_proj_perf_dict, index=[0]).to_csv("{}seed_{}_{}_spar_{}.csv".format(artifact_path, args.seed, args.mixtype, args.spar_alpha))
        pd.DataFrame(ols_test_perf_dict, index=[0]).to_csv("{}seed_{}_{}_ols.csv".format(artifact_path, args.seed, args.mixtype))


        # write results
        write_result(args, args.kde_bandwidth, result_dict_best, result_path)
        if args.dataset in ['CommunitiesAndCrime', 'SkillCraft']:
            print(model.fclayer)
        elif args.dataset in ['RCF_MNIST', 'ChairAngle_Tails']:
            print(model.fc)
        if args.store_model:
            if args.dataset == 'ChairAngle_Tails':
                print("Can't save TMLR models.")
            else:
                write_model(args, best_model_dict[args.metrics], result_path)

    else: # use best model, 1 for rmse or 2 for r
        assert args.read_best_model == 1
        # extra_str = '' if args.metrics == 'rmse' else 'r'
        pt_full_path = result_path + get_unique_file_name(args, '','.pickle')
        
        with open(pt_full_path,'rb') as f:
            s = f.read()
            read_model = pickle.loads(s)
        print('load best model success from {pt_full_path}!')

        all_begin = time.time()
        
        print('='*30 + ' read best model and verify result ' + '=' * 30)
        read_result_dic = algorithm.test(args, read_model, data_packet['x_test'], data_packet['y_test'],
                        ('seed = ' + str(args.seed) + ': Final test for read model: ' + pt_full_path + ':\n'),
                        True, all_begin,  device)            
                        
        algorithm.cal_worst_acc(args,data_packet,read_model,read_result_dic,all_begin,ts_data, device)
        
        write_result(args, 'read', read_result_dic, result_path, '') # rewrite result txt

if __name__ == '__main__':
    main()
