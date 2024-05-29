import argparse
import copy
import datetime
import random
import json
import os
import sys
import csv
from collections import defaultdict
from tempfile import mkdtemp

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import chi2
import scipy
import pandas as pd

import wilds
import models
from config import dataset_defaults
from utils import unpack_data, save_best_model, \
    Logger, return_predict_fn, return_criterion, save_pred

from mixup import mix_up


def save_reps_and_labels(args, model, data_loader):

    # Create a copy of the model, and swap out the classification
    # layer for an identity.
    model_copy = copy.deepcopy(model)
    model_copy.enc.fc = torch.nn.Identity() 
    model_copy.eval()
    representations, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            representation = model_copy(x).detach().cpu().numpy()
            ys.append(y.detach().cpu().numpy())
            representations.append(representation)
            metas.append(batch[2].detach().cpu().numpy())

        return np.concatenate(representations), np.concatenate(ys), np.concatenate(metas)


def save_reps_only(args, model, data_loader):
    # this *should* work with both labeled and unlabeled data loaders...

    # Create a copy of the model, and swap out the classification
    # layer for an identity.
    model_copy = copy.deepcopy(model)
    model_copy.enc.fc = torch.nn.Identity() 
    model_copy.eval()
    representations, metas = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            print(i)
            # get the inputs
            x = batch[0]
            x = x.to(device)
            metadata = batch[-1]
            representation = model_copy(x).detach().cpu().numpy()
            representations.append(representation)
            metas.append(metadata.detach().cpu().numpy())

        return np.concatenate(representations), np.concatenate(metas)


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


def save_eigenmetric(X_train_representations, Z_test_representations, Y_train_labels, sigma_squared, 
                     save_metric_matrix_path):
    """Save the matrix of evalue ratios and evector dot prods.

    Parameters:
        X_train_representations: torch tensor. The training representations.
        Z_test_representations: torch tensor. The test representations.
        Y_train_labels: torch tensor. The training labels.
        sigma_squared: float. The estimated variance of the label noise.
        save_metric_matrix_path: str. Where to save the matrix.

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

    # Get the test eigenvector variances
    eig_correlations = np.matmul(vh_x, vh_z.transpose())
    eigenratio_matrix = np.matmul((1/squared_s_x)[:, None], squared_s_z[None, :])
    eigenmetric_matrix = (eig_correlations**2)*eigenratio_matrix
    # save a normalized version of the eigenmetric matrix:
    normalized_squared_s_z = (s_z/np.sqrt(Z_test_representations.shape[0]))**2
    normalized_eigenratio_matrix = np.matmul((1/squared_s_x)[:, None], normalized_squared_s_z[None, :])
    normalized_eigenmetric_matrix = (eig_correlations**2)*normalized_eigenratio_matrix
    np.save(save_metric_matrix_path, normalized_eigenmetric_matrix)




# code base: https://github.com/huaxiuyao/LISA/tree/main/domain_shifts

runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Povertymap experiments for SpAR.')
# General
parser.add_argument('--dataset', type=str, default='poverty',
                    help="Name of dataset")
parser.add_argument('--algorithm', type=str, default='erm',
                    help='training scheme, choose between fish or erm.')
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--experiment_dir', type=str, default='../',
                    help='experiment directory')
parser.add_argument('--data-dir', type=str, default='./',
                    help='path to data dir')

# SpAR stuff
parser.add_argument("--projection", default=False, action='store_true')
parser.add_argument('--base_model_path', type=str, default='./',
                    help='path to model we are adapting')
parser.add_argument('--spar_alpha', type=float, default=0.999,
                    help='The confidence value for SpAR')
parser.add_argument('--proj_artifact_dir', type=str, default='./',
                    help='The directory that we will save artifacts in')
parser.add_argument('--search_lr', type=float, default=None,
                    help="The new learning rate we are trying in hyperparam search")
parser.add_argument('--search_artifact_base_path', type=str, default=None,
                    help="The base path for artifacts from this param setting")
parser.add_argument("--use_bias", default=False, action='store_true')
# Computation
parser.add_argument('--nocuda', type=int, default=False,
                    help='disables CUDA use')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed, set as -1 for random.')
parser.add_argument("--mix_alpha", default=0.5, type=float)
parser.add_argument("--print_loss_iters", default=100, type=int)
parser.add_argument("--kde_bandwidth", default=0.5, type=float)

# Whether we want to save the eigenmetric matrix for plotting later (figure 2)
parser.add_argument("--save_eigenmetric_matrix", default=False, action='store_true')

parser.add_argument("--is_kde", default=0, type=int) # kde mixup or random mixup
parser.add_argument("--save_pred", default=False, action='store_true')
parser.add_argument("--save_dir", default='result', type=str)
parser.add_argument("--fold", default='A', type=str)
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers that the DataLoader will create')
parser.add_argument("--adapt_to_unlabeled_test_data", default=False, action='store_true')
# DARE-GRAM hyparams
parser.add_argument('--dare_gram_tradeoff_angle', type=float, default=0.05,
                        help='tradeoff for angle alignment')
parser.add_argument('--dare_gram_tradeoff_scale', type=float, default=0.001,
                        help='tradeoff for scale alignment')
parser.add_argument('--dare_gram_treshold', type=float, default=0.9,
                        help='treshold for the pseudo inverse')
# RSD hyparams
parser.add_argument('--rsd_tradeoff', type=float, default=0.001,
                        help='tradeoff of RSD')
parser.add_argument('--rsd_tradeoff2', type=float, default=0.01,
                        help='tradeoff of BMP')




args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda")
if args.nocuda:
    print(f'use cpu')
    device = torch.device("cpu")

args_dict = args.__dict__
args_dict.update(dataset_defaults[args.dataset]) # default configuration
# Swap out the learning rate with the search LR if its been specified
if args_dict['search_lr'] is not None:
    args_dict['optimiser_args']['lr'] = args_dict['search_lr']
args = argparse.Namespace(**args_dict)

# random select a training fold according to seed. Can comment this line and set args.fold manually as well
args.fold = ['A', 'B', 'C', 'D', 'E'][args.seed % 5] 

if args.save_pred:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

##### set seed #####
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
print(f'args.seed = {args.seed}')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# experiment directory setup
args.experiment = f"{args.dataset}_{args.algorithm}_{args.seed}" \
    if args.experiment == '.' else args.experiment
if args.is_kde:
    args.experiment += f'_kde_bw{args.kde_bandwidth}'
directory_name = '{}/experiments/{}'.format(args.experiment_dir,args.experiment)

if not os.path.exists(directory_name):
    os.makedirs(directory_name)
runPath = mkdtemp(prefix=runId, dir=directory_name)

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
print(args)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

# load model
modelC = getattr(models, args.dataset)
if args.algorithm == 'mixup': args.batch_size //= 2

train_loader, tv_loaders, unlabeled_loaders = modelC.getDataLoaders(args, device=device)
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']
model = modelC(args, weights=None).to(device)

print(f'len(train_loader) = {len(train_loader)}, len(val_loader) = {len(val_loader)}, len(test_loader) = {len(test_loader)}')
print(f'len(unlabeled_test_loader) = {len(unlabeled_loaders["test_unlabeled"])}')


n_class = getattr(models, f"{args.dataset}_n_class")

assert args.optimiser in ['SGD', 'Adam', 'AdamW'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
opt = getattr(optim, args.optimiser)

params = filter(lambda p: p.requires_grad, model.parameters())
optimiserC = opt(params, **args.optimiser_args)

predict_fn, criterion = return_predict_fn(args.dataset), return_criterion(args.dataset)

def train_erm(train_loader, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} ,arg = erm'.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        optimiserC.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        #if args.use_bert_params:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(),
        #                                   args.max_grad_norm)
        optimiserC.step()

        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0 and args.print_iters != -1 :
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print(
                'iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg, args)


def train_dare_gram(train_loader, epoch, agg,
                    uda_loader=unlabeled_loaders['test_unlabeled']):
    # NOTE: uda_loader is the "unsupervised domain adaptation" loader, which 
    #       defaults to the unlabeled test data defined above. This will be used
    #       to compute the DARE-GRAM loss.

    def DARE_GRAM_LOSS(H1, H2):    
        """https://github.com/ismailnejjar/DARE-GRAM/blob/main/code/dSprites/dare_gram.py"""
        b,p = H1.shape

        A = torch.cat((torch.ones(b,1).to(device), H1), 1)
        B = torch.cat((torch.ones(b,1).to(device), H2), 1)

        cov_A = (A.t()@A)
        cov_B = (B.t()@B) 

        _,L_A,_ = torch.linalg.svd(cov_A)
        _,L_B,_ = torch.linalg.svd(cov_B)
        
        eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
        eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

        if(eigen_A[1]>args.dare_gram_treshold):
            T = eigen_A[1].detach()
        else:
            T = args.dare_gram_treshold
            
        index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

        if(eigen_B[1]>args.dare_gram_treshold):
            T = eigen_B[1].detach()
        else:
            T = args.dare_gram_treshold

        index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
        
        k = max(index_A, index_B)[0]

        A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
        B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
        
        cos_sim = torch.nn.CosineSimilarity(dim=0,eps=1e-6)
        cos = torch.dist(torch.ones((p+1)).to(device),(cos_sim(A,B)),p=1)/(p+1)
        
        penalty1 = cos # the "angle" penalty
        penalty2 = torch.dist((L_A[:k]),(L_B[:k]))/k # the "scale" penalty
        return penalty1, penalty2


    running_mse = 0
    running_penalty1 = 0
    running_penalty2 = 0
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} ,arg = dare_gram'.format(epoch))
    uda_batches = iter(uda_loader) # so I can grab the next batch
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        try:
            # get a batch of unlabeled data from target domain
            uda_batch = next(uda_batches)
            x_target = uda_batch[0].to(device)
            # trim to same size as source batch
            x_target = x_target[:len(x)]
            assert len(x_target) == len(x), 'Target batch was small while source batch was big'
        except StopIteration: # ran out of UDA data
            uda_batches = iter(uda_loader)
            uda_batch = next(uda_batches)
            x_target = uda_batch[0].to(device)
        except AssertionError: # this UDA batch isn't full sized; skip it
            uda_batches = iter(uda_loader)
            uda_batch = next(uda_batches)
            x_target = uda_batch[0].to(device)
        optimiserC.zero_grad()
        # y_hat = model(x)
        y_hat, z = model.enc(x, with_feats=True)
        _, z_target = model.enc(x_target, with_feats=True)
        penalty1, penalty2 = DARE_GRAM_LOSS(z, z_target)
        running_penalty1 += penalty1.clone().item()
        running_penalty2 += penalty2.clone().item()
        total_dare_gram_penalty = args.dare_gram_tradeoff_angle * penalty1 \
                            + args.dare_gram_tradeoff_scale * penalty2
        mse = criterion(y_hat, y)
        running_mse += mse.clone().item()
        loss = mse + total_dare_gram_penalty
        loss.backward()
        #if args.use_bert_params:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(),
        #                                   args.max_grad_norm)
        optimiserC.step()

        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0 and args.print_iters != -1 :

            agg['train_mse'].append(running_mse / args.print_loss_iters)
            agg['train_penalty1'].append(running_penalty1 / args.print_loss_iters)
            agg['train_penalty2'].append(running_penalty2 / args.print_loss_iters)
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print(
                'iteration {:05d}/{:d}: MSE: {:6.3f}, penalty1: {:6.3f}, penalty1: {:6.3f}, loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters,
                running_penalty1 / args.print_loss_iters,
                running_penalty2 / args.print_loss_iters,
                running_loss / args.print_loss_iters
                ))
            running_mse = 0.0
            running_penalty1 = 0.0
            running_penalty2 = 0.0
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg, args)


def train_rsd(train_loader, epoch, agg,
              uda_loader=unlabeled_loaders['test_unlabeled']):
    # NOTE: uda_loader is the "unsupervised domain adaptation" loader, which 
    #       defaults to the unlabeled test data defined above. This will be used
    #       to compute the RSD loss.

    def RSD(Feature_s, Feature_t):
        """https://github.com/thuml/Domain-Adaptation-Regression/blob/master/DAR-RSD/dSprites/train_rsd.py"""
        u_s, s_s, v_s = torch.svd(Feature_s.t())
        u_t, s_t, v_t = torch.svd(Feature_t.t())
        p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
        sinpa = torch.sqrt(1-torch.pow(cospa,2))
        # return torch.norm(sinpa,1) + args.tradeoff2*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)
        penalty1 = torch.norm(sinpa,1) 
        penalty2 = torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)
        return penalty1, penalty2


    running_mse = 0
    running_penalty1 = 0
    running_penalty2 = 0
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} ,arg = rsd'.format(epoch))
    uda_batches = iter(uda_loader) # so I can grab the next batch
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        try:
            # get a batch of unlabeled data from target domain
            uda_batch = next(uda_batches)
            x_target = uda_batch[0].to(device)
            # trim to same size as source batch
            x_target = x_target[:len(x)]
            assert len(x_target) == len(x), 'Target batch was small while source batch was big'
        except StopIteration: # ran out of UDA data
            uda_batches = iter(uda_loader)
            uda_batch = next(uda_batches)
        except AssertionError: # this UDA batch isn't full sized; skip it
            uda_batches = iter(uda_loader)
            uda_batch = next(uda_batches)
        optimiserC.zero_grad()
        # y_hat = model(x)
        y_hat, z = model.enc(x, with_feats=True)
        _, z_target = model.enc(x_target, with_feats=True)
        penalty1, penalty2 = RSD(z, z_target)
        running_penalty1 += penalty1.clone().item()
        running_penalty2 += penalty2.clone().item()
        total_rsd_penalty = args.rsd_tradeoff * (
            penalty1 + args.rsd_tradeoff2 * penalty2
        )
        mse = criterion(y_hat, y)
        running_mse += mse.clone().item()
        loss = mse + total_rsd_penalty
        loss.backward()
        #if args.use_bert_params:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(),
        #                                   args.max_grad_norm)
        optimiserC.step()

        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0 and args.print_iters != -1 :
            agg['train_mse'].append(running_mse / args.print_loss_iters)
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_penalty1'].append(running_penalty1 / args.print_loss_iters)
            agg['train_penalty2'].append(running_penalty2 / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print(
                'iteration {:05d}/{:d}: MSE: {:6.3f}, penalty1: {:6.3f}, penalty1: {:6.3f}, loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters,
                running_penalty1 / args.print_loss_iters,
                running_penalty2 / args.print_loss_iters,
                running_loss / args.print_loss_iters
                ))
            running_mse = 0.0
            running_penalty1 = 0.0
            running_penalty2 = 0.0
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg, args)




def train_mixup(train_loader, epoch, agg):
    print('into train_mixup')
    model.train()
    train_loader.dataset.reset_batch()
    print('\n====> Epoch: {:03d} '.format(epoch))

    # The probabilities for each group do not equal to each other.
    for i, data in enumerate(train_loader):
        model.train()
        x1, y1, g1, prev_idx = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        if y1.ndim > 1:
            y1 = y1.squeeze()
        x2, y2, g2 = [], [], []
        
        assert train_loader.dataset.num_envs > 1
        for g, y, idx in zip(g1,y1, prev_idx):
            tmp_x, tmp_y, tmp_g = train_loader.dataset.get_sample(idx.item(), UseKDE = args.is_kde, y1=y)
            x2.append(tmp_x.unsqueeze(0))
            y2.append(tmp_y)
            g2.append(tmp_g)

        x2 = torch.cat(x2).to(device)
        y2 = torch.cat(y2).to(device)

        loss_fn = torch.nn.MSELoss()
        # mixup
        mixed_x1, mixed_y1 = mix_up(args, x1, y1, x2, y2, args.dataset)
        mixed_x2, mixed_y2 = mix_up(args, x2, y2, x1, y1, args.dataset)

        mixed_x = torch.cat([mixed_x1, mixed_x2])
        mixed_y = torch.cat([mixed_y1, mixed_y2])

        # forward
        outputs = model(mixed_x)

        loss = loss_fn(outputs, mixed_y)

        # backward
        optimiserC.zero_grad()
        loss.backward()
        optimiserC.step()

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            print(f'iteration {(i + 1):05d}: ')
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg, args)
            model.train()

def test(test_loader, agg, loader_type='test', verbose=True, save_ypred=False, save_dir=None, return_dict=False):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(batch[2])

        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        if save_ypred: # random select a fold
            save_name = f"{args.dataset}_split:{loader_type}_fold:" \
                        f"{['A', 'B', 'C', 'D', 'E'][args.seed % 5]}" \
                        f"_epoch:best_pred.csv"
            with open(f"{runPath}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        print("Test metas shape is {}".format(metas.shape))
        test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)

        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")
        if return_dict:
            return test_val


def linear_test(eval_loader, test_representations, test_labels, test_metas, linear_model, agg, is_train=False, save_pred_path=None, save_target_path=None):
    linear_model.eval()
    with torch.no_grad():
        # get the inputs
        x, y = torch.from_numpy(test_representations).to(device), torch.from_numpy(test_labels).to(device)
        y_hat = linear_model(x)


        ypreds = predict_fn(y_hat)
        if is_train:
            total_se = (np.square(ypreds.cpu().numpy()  - y.cpu().numpy())).sum()
            return total_se

        else:
            test_val = eval_loader.dataset.eval(ypreds.cpu(), y.cpu(), torch.from_numpy(test_metas).cpu())

            if save_pred_path is not None and save_target_path is not None:
                print("Preds shape {}".format(ypreds.shape))
                print("Value shape {}".format(y.shape))
                pd.DataFrame(ypreds.cpu().numpy(), columns=['Preds']).to_csv(save_pred_path, index=False, header=False)
                pd.DataFrame(y.cpu().numpy(), columns=['Targets']).to_csv(save_target_path, index=False, header=False)

            print(test_val)
            print(f"=============== {is_train} ===============\n{test_val[-1]}")
            return test_val


if __name__ == '__main__':
    # set learning rate schedule
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimiserC,step_size = 1, **args.scheduler_kwargs)
        scheduler.step_every_batch = False
        scheduler.use_metric = False
    else:
        scheduler = None

    print("=" * 30 + f" Training: {args.algorithm} for {args.dataset} "  + "=" * 30)
    
    train = locals()[f'train_{args.algorithm}'] 
    agg = defaultdict(list)
    agg['val_stat'] = [0.]
    agg['test_stat'] = [0.]

    if not args.projection:

        for epoch in range(args.epochs):
            train(train_loader, epoch, agg)
            test(val_loader, agg,'val',True)
            if scheduler is not None:
                scheduler.step()

            test(test_loader, agg,'test', True)
            save_best_model(model, runPath, agg, args)

            if args.save_pred:
                save_pred(args,model, train_loader, epoch, args.save_dir,predict_fn,device)

        model.load_state_dict(torch.load(runPath + '/model.rar'))
        print('Finished training! Loading best model...')
        perf_dicts_and_names = []
        for split, loader in tv_loaders.items():
            perf_dict = test(loader, agg, loader_type=split,verbose=True, save_ypred=True, return_dict=True)[0]
            perf_dicts_and_names.append((split, perf_dict))
        
        # Compile together all the performance dictionaries and save them
        full_performance_dictionary = {}
        for split_name, perf_dict in perf_dicts_and_names:
            for key, value in perf_dict.items():
                full_performance_dictionary['{}_{}'.format(split_name, key)] = value

        artifact_path = args.search_artifact_base_path
        full_performance_dictionary['seed'] = args.seed
        full_performance_dictionary['method'] = args.algorithm
        full_performance_dictionary['lr'] = args.optimiser_args['lr']
        full_performance_dictionary['bw'] = args.kde_bandwidth
        pd.DataFrame(full_performance_dictionary, index=[0]).to_csv("{}.csv".format(artifact_path, args.seed, args.algorithm))
 
    if args.projection:
        if not os.path.exists(args.proj_artifact_dir):
            os.mkdir(args.proj_artifact_dir)
        # Load in the model
        model.load_state_dict(torch.load("{}model.rar".format(args.base_model_path)))
        
        original_test_performance_dict = test(test_loader, agg, loader_type='test', verbose=True, save_ypred=True, return_dict=True)[0]
        original_test_full_performance_dictionary = {}
        for key, value in original_test_performance_dict.items():
            original_test_full_performance_dictionary['test_{}'.format(key)] = value
        original_test_full_performance_dictionary['seed'] = args.seed
        original_test_full_performance_dictionary['method'] = args.algorithm
        original_test_full_performance_dictionary['lr'] = args.optimiser_args['lr']
        original_test_full_performance_dictionary['bw'] = args.kde_bandwidth
        pd.DataFrame(original_test_full_performance_dictionary, index=[0]).to_csv("{}/original_perf.csv".format(args.proj_artifact_dir))


        # Gather representations for each of the datasets
        print(model)
        print('computing representations on training data')
        train_reps, train_ys, train_metas = save_reps_and_labels(args, model, train_loader)
        print('computing representations on validation data')
        val_reps, val_ys, val_metas = save_reps_and_labels(args, model, val_loader)
        print('computing representations on test data')
        test_reps, test_ys, test_metas = save_reps_and_labels(args, model, test_loader)
        id_test_reps, id_test_ys, id_test_metas = save_reps_and_labels(
            args, model, tv_loaders['id_test']
        )
        id_val_reps, id_val_ys, id_val_metas = save_reps_and_labels(
            args, model, tv_loaders['id_val']
        )
        if args.adapt_to_unlabeled_test_data:
            print(
                'computing representations on extra unlabeled train/valid/test data'
            )
            unlabeled_test_reps, unlabeled_test_metas = save_reps_only(
                args, model, unlabeled_loaders['test_unlabeled']
            )
        print("TRAIN SHAPES")
        print(train_reps.shape)
        print(train_ys.shape)
        print(train_metas.shape)

        print("VALIDATION SHAPES")
        print(val_reps.shape)
        print(val_ys.shape)
        print(val_metas.shape)

        print("ID TEST VS OOD TEST SHAPES")
        print(id_test_reps.shape)
        print(test_reps.shape)

        if args.adapt_to_unlabeled_test_data:
            print("UNLABELED TEST SHAPES")
            print(unlabeled_test_reps.shape)
            print(unlabeled_test_metas.shape)

        # Test the original model's performance with our linear prediction scheme
        og_model = torch.nn.Linear(512, 1, bias=False)
        og_model = og_model.to(device)
        new_state_dict = {'weight': torch.from_numpy(model.enc.fc.weight.detach().cpu().numpy())}
        og_model.load_state_dict(new_state_dict)
        og_model = og_model.to(device)

        og_test_performance = linear_test(test_loader, test_reps, test_ys, test_metas, og_model, agg, is_train=False)[0]

        print(original_test_performance_dict)
        print(og_test_performance)


        # Estimate sigma^2 using the performance of the OLS soln
        train_pinv_soln_for_sigma = compute_pseudoinvese_soln(train_reps, train_ys)
        ols_model = torch.nn.Linear(512, 1, bias=False)
        ols_model = ols_model.to(device)
        new_state_dict = {'weight': torch.from_numpy(train_pinv_soln_for_sigma).transpose(1, 0)}
        ols_model.load_state_dict(new_state_dict)
        ols_model = ols_model.to(device)

        ols_error = linear_test(train_loader, train_reps, train_ys, train_metas, ols_model, agg, is_train=True)
        print(ols_error)
        sigma_squared_estimate = ols_error/train_reps.shape[0]
        print(sigma_squared_estimate)

        # Get the performance of OLS on the sets
        print("OLS model performance")
        ols_val_performance = linear_test(val_loader, val_reps, val_ys, val_metas, ols_model, agg, is_train=False)
        ols_test_performance = linear_test(test_loader, test_reps, test_ys, test_metas, ols_model, agg, is_train=False)
        ols_id_val_performance = linear_test(tv_loaders['id_val'], id_val_reps, id_val_ys, id_val_metas, ols_model, agg, is_train=False)
        ols_id_test_performance = linear_test(tv_loaders['id_test'], id_test_reps, id_test_ys, id_test_metas, ols_model, agg, is_train=False)



        # Save the results as a CSV
        ols_test_perf_dict = ols_test_performance[0]
        ols_val_perf_dict = ols_val_performance[0]

        ols_full_performance_dictionary = {}
        for split_name, perf_dict in [('test', ols_test_perf_dict), ('val', ols_val_perf_dict)]:
            for key, value in perf_dict.items():
                ols_full_performance_dictionary['{}_{}'.format(split_name, key)] = value

        ols_full_performance_dictionary['seed'] = args.seed
        ols_full_performance_dictionary['method'] = "{}_OLS".format(args.algorithm)
        ols_full_performance_dictionary['spar_alpha'] = args.spar_alpha
        ols_full_performance_dictionary['lr'] = args.optimiser_args['lr']
        ols_full_performance_dictionary['bw'] = args.kde_bandwidth
        pd.DataFrame(ols_full_performance_dictionary, index=[0]).to_csv("{}/ols_perf.csv".format(args.proj_artifact_dir))

        if args.save_eigenmetric_matrix:

            # Save ID test set eigenmetric matrix
            save_eigenmetric(
                train_reps, id_test_reps, train_ys,
                sigma_squared=sigma_squared_estimate,
                save_metric_matrix_path="{}/povertymap_{}_lr_{}_bw_{}_seed_{}_normalized_id_test_set_eigenmetric".format(
                    args.proj_artifact_dir,
                    args.algorithm,
                    args.optimiser_args['lr'],
                    args.kde_bandwidth,
                    args.seed
                )
            )

        # now adapt the regressor we actually want to evaluate OOD
        if args.adapt_to_unlabeled_test_data:
            target_reps = unlabeled_test_reps
        else:
            target_reps = test_reps
    
        if args.save_eigenmetric_matrix:
            # Save OOD test set eigenmetric matrix
            if args.adapt_to_unlabeled_test_data:
                current_save_metric_matrix_path = "{}/povertymap_{}_lr_{}_bw_{}_seed_{}_normalized_UDA_test_set_eigenmetric".format(
                    args.proj_artifact_dir,
                    args.algorithm,
                    args.optimiser_args['lr'],
                    args.kde_bandwidth,
                    args.seed
                )
            else:
                current_save_metric_matrix_path = "{}/povertymap_{}_lr_{}_bw_{}_seed_{}_normalized_transductive_test_set_eigenmetric".format(
                    args.proj_artifact_dir,
                    args.algorithm,
                    args.optimiser_args['lr'],
                    args.kde_bandwidth,
                    args.seed
                )
            save_eigenmetric(
            train_reps, target_reps, train_ys,
            sigma_squared=sigma_squared_estimate, 
            save_metric_matrix_path=current_save_metric_matrix_path
            )

        w_proj = spar_chi_adaptation(
            train_reps, target_reps, train_ys,
            sigma_squared=sigma_squared_estimate
        )

        w_proj_model = torch.nn.Linear(512, 1, bias=False)
        w_proj_model = w_proj_model.to(device)
        new_state_dict = {'weight': torch.from_numpy(w_proj).transpose(1, 0)}
        w_proj_model.load_state_dict(new_state_dict)
        w_proj_model = w_proj_model.to(device)

        print("Projection model performance")
        os.mkdir("{}/w_proj_predictions".format(args.proj_artifact_dir))
        os.mkdir("{}/w_proj_targets".format(args.proj_artifact_dir))
        w_proj_val_performance = linear_test(val_loader, val_reps, val_ys, val_metas, w_proj_model, agg, is_train=False,
                save_pred_path="{}/w_proj_predictions/poverty_split:val_fold:{}_epoch:best_pred.csv".format(args.proj_artifact_dir, args.fold), save_target_path='{}/w_proj_targets/poverty_split:val_fold:{}_epoch:targets.csv'.format(args.proj_artifact_dir, args.fold))
        w_proj_test_performance = linear_test(test_loader, test_reps, test_ys, test_metas, w_proj_model, agg, is_train=False,
                save_pred_path="{}/w_proj_predictions/poverty_split:test_fold:{}_epoch:best_pred.csv".format(args.proj_artifact_dir, args.fold), save_target_path='{}/w_proj_targets/poverty_split:test_fold:{}_epoch:targets.csv'.format(args.proj_artifact_dir, args.fold))
        w_proj_id_val_performance = linear_test(tv_loaders['id_val'], id_val_reps, id_val_ys, id_val_metas, w_proj_model, agg, is_train=False,
                save_pred_path="{}/w_proj_predictions/poverty_split:id_val_fold:{}_epoch:best_pred.csv".format(args.proj_artifact_dir, args.fold), save_target_path='{}/w_proj_targets/poverty_split:id_val_fold:{}_epoch:targets.csv'.format(args.proj_artifact_dir, args.fold))
        w_proj_id_test_performance = linear_test(tv_loaders['id_test'], id_test_reps, id_test_ys, id_test_metas, w_proj_model, agg, is_train=False,
                save_pred_path="{}/w_proj_predictions/poverty_split:id_test_fold:{}_epoch:best_pred.csv".format(args.proj_artifact_dir, args.fold), save_target_path='{}/w_proj_targets/poverty_split:id_test_fold:{}_epoch:targets.csv'.format(args.proj_artifact_dir, args.fold))



        # Save the results as a CSV
        w_proj_test_perf_dict = w_proj_test_performance[0]
        w_proj_val_perf_dict = w_proj_val_performance[0]

        full_performance_dictionary = {}
        for split_name, perf_dict in [('test', w_proj_test_perf_dict), ('val', w_proj_val_perf_dict)]:
            for key, value in perf_dict.items():
                full_performance_dictionary['{}_{}'.format(split_name, key)] = value

        full_performance_dictionary['seed'] = args.seed
        full_performance_dictionary['method'] = "{}_SPAR".format(args.algorithm)
        full_performance_dictionary['spar_alpha'] = args.spar_alpha
        full_performance_dictionary['lr'] = args.optimiser_args['lr']
        full_performance_dictionary['bw'] = args.kde_bandwidth
        pd.DataFrame(full_performance_dictionary, index=[0]).to_csv("{}/w_proj_perf.csv".format(args.proj_artifact_dir))


    print('done')
