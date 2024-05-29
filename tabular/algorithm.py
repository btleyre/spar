import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import time
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KernelDensity
from utils import stats_values


def cal_worst_acc(args,data_packet,best_model_rmse,best_result_dict_rmse,all_begin,ts_data,device):
    #### worst group acc ---> rmse ####
    if args.is_ood:
        x_test_assay_list = data_packet['x_test_assay_list']
        y_test_assay_list = data_packet['y_test_assay_list']
        worst_acc = 0.0 if args.metrics == 'rmse' else 1e10
            
        for i in range(len(x_test_assay_list)):
            result_dic = test(args,best_model_rmse,x_test_assay_list[i],y_test_assay_list[i],
                            '', False, all_begin, device)
            acc = result_dic[args.metrics] 
            if args.metrics == 'rmse':
                if acc > worst_acc:
                    worst_acc = acc
            else:#r
                if np.abs(acc) < np.abs(worst_acc):
                    worst_acc = acc
        print('worst {} = {:.3f}'.format(args.metrics, worst_acc))
        best_result_dict_rmse['worst_' + args.metrics] = worst_acc


def linear_cal_worst_acc(args,data_packet,best_model_rmse, best_result_dict_rmse,all_begin,ts_data,device, predictor_model):
    #### worst group acc ---> rmse ####
    if args.is_ood:
        x_test_assay_list = data_packet['x_test_assay_list']
        y_test_assay_list = data_packet['y_test_assay_list']
        worst_acc = 0.0 if args.metrics == 'rmse' else 1e10
            
        for i in range(len(x_test_assay_list)):
            x_assay_reps, y_assay_labels = save_reps_and_labels(args, best_model_rmse, x_test_assay_list[i], y_test_assay_list[i], device)
            result_dic = linear_model_test(args, predictor_model, x_assay_reps, y_assay_labels,
                            '', False, all_begin, device)
            acc = result_dic[args.metrics] 
            if args.metrics == 'rmse':
                if acc > worst_acc:
                    worst_acc = acc
            else:#r
                if np.abs(acc) < np.abs(worst_acc):
                    worst_acc = acc
        print('worst {} = {:.3f}'.format(args.metrics, worst_acc))
        best_result_dict_rmse['worst_' + args.metrics] = worst_acc


def get_mixup_sample_rate(args, data_packet, device='cuda', use_kde = False):
    
    mix_idx = []
    _, y_list = data_packet['x_train'], data_packet['y_train'] 
    is_np = isinstance(y_list,np.ndarray)
    if is_np:
        data_list = torch.tensor(y_list, dtype=torch.float32)
    else:
        data_list = y_list

    N = len(data_list)

    ######## use kde rate or uniform rate #######
    for i in range(N):
        if args.mixtype == 'kde' or use_kde: # kde
            data_i = data_list[i]
            #print(data_list)
            #print(data_list.shape)
            #print(data_i)
            #print(data_i.shape)
            data_i = data_i.reshape(-1,data_i.shape[0]) # get 2D
            
            if args.show_process:
                if i % (N // 10) == 0:
                    print('Mixup sample prepare {:.2f}%'.format(i * 100.0 / N ))
                
            ######### get kde sample rate ##########
            kd = KernelDensity(kernel=args.kde_type, bandwidth=args.kde_bandwidth).fit(data_i)  # should be 2D
            each_rate = np.exp(kd.score_samples(data_list))
            each_rate /= np.sum(each_rate)  # norm
        else:
            each_rate = np.ones(y_list.shape[0]) * 1.0 / y_list.shape[0]
        
        ####### visualization: observe relative rate distribution shot #######
        if args.show_process and i == 0:
                print(f'bw = {args.kde_bandwidth}')
                print(f'each_rate[:10] = {each_rate[:10]}')
                stats_values(each_rate)
            
        mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)

    self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]

    if args. show_process:
        print(f'len(y_list) = {len(y_list)}, len(mix_idx) = {len(mix_idx)}, np.mean(self_rate) = {np.mean(self_rate)}, np.std(self_rate) = {np.std(self_rate)},  np.min(self_rate) = {np.min(self_rate)}, np.max(self_rate) = {np.max(self_rate)}')

    return mix_idx


def get_batch_kde_mixup_idx(args, Batch_X, Batch_Y, device):
    assert Batch_X.shape[0] % 2 == 0
    Batch_packet = {}
    Batch_packet['x_train'] = Batch_X.cpu()
    Batch_packet['y_train'] = Batch_Y.cpu()

    Batch_rate = get_mixup_sample_rate(args, Batch_packet, device, use_kde=True) # batch -> kde
    if args. show_process:
        stats_values(Batch_rate[0])
    idx2 = [np.random.choice(np.arange(Batch_X.shape[0]), p=Batch_rate[sel_idx]) 
            for sel_idx in np.arange(Batch_X.shape[0]//2)]
    return idx2

def get_batch_kde_mixup_batch(args, Batch_X1, Batch_X2, Batch_Y1, Batch_Y2, device):
    Batch_X = torch.cat([Batch_X1, Batch_X2], dim = 0)
    Batch_Y = torch.cat([Batch_Y1, Batch_Y2], dim = 0)

    idx2 = get_batch_kde_mixup_idx(args,Batch_X,Batch_Y,device)

    New_Batch_X2 = Batch_X[idx2]
    New_Batch_Y2 = Batch_Y[idx2]
    return New_Batch_X2, New_Batch_Y2


def linear_model_test(args, model, x_list, y_list, name, need_verbose, epoch_start_time, device):
    model.eval()
    with torch.no_grad():

        if not isinstance(y_list, np.ndarray):
            y_list = y_list.numpy()
        if not isinstance(x_list, np.ndarray):
            x_list = x_list.numpy()

        x_list = torch.from_numpy(x_list).to(device)
        

        model = model.to(device)
        pred_y = model(x_list).cpu().numpy()
        y_list = y_list.squeeze()
        y_list_pred = pred_y.squeeze()
        
        ###### calculate metrics ######

        mean_p = y_list_pred.mean(axis = 0)
        sigma_p = y_list_pred.std(axis = 0)
        mean_g = y_list.mean(axis = 0)
        sigma_g = y_list.std(axis = 0)

        index = (sigma_g!=0)
        corr = ((y_list_pred - mean_p) * (y_list - mean_g)).mean(axis = 0) / (sigma_p * sigma_g)
        corr = (corr[index]).mean()

        mse = (np.square(y_list_pred  - y_list )).mean()
        total_se = (np.square(y_list_pred  - y_list )).sum()
        result_dict = {'mse':mse, 'r':corr, 'r^2':corr**2, 'rmse':np.sqrt(mse), 'total_se': total_se}

        not_zero_idx = y_list != 0.0
        mape = (np.fabs(y_list_pred[not_zero_idx] -  y_list[not_zero_idx]) / np.fabs(y_list[not_zero_idx])).mean() * 100
        result_dict['mape'] = mape
        
    ### verbose ###
    if need_verbose:
        epoch_use_time = time.time() - epoch_start_time
        # valid -> interval time; final test -> all time
        print(name + 'corr = {:.4f}, rmse = {:.4f}, mape = {:.4f} %'.format(corr,np.sqrt(mse),mape) + ', time = {:.4f} s'.format(epoch_use_time))
        
    return result_dict



def test(args, model, x_list, y_list, name, need_verbose, epoch_start_time, device):
    model.eval()
    with torch.no_grad():
        if args.dataset == 'Dti_dg': 
            val_iter = x_list.shape[0] // args.batch_size 
            val_len = args.batch_size
            y_list = y_list[:val_iter * val_len]
        elif args.dataset == "ChairAngle_Tails":
            val_iter = 2
            val_len = int(np.ceil(x_list.shape[0]/2))            
        else: # read in the whole test data
            val_iter = 1
            val_len = x_list.shape[0]
        y_list_pred = []
        assert val_iter >= 1 #  easy test

        for ith in range(val_iter):
            if isinstance(x_list,np.ndarray):
                x_list_torch = torch.tensor(x_list[ith*val_len:(ith+1)*val_len], dtype=torch.float32).to(device)
            else:
                x_list_torch = x_list[ith*val_len:(ith+1)*val_len].to(device)

            model = model.to(device)
            pred_y = model(x_list_torch).cpu().numpy()
            y_list_pred.append(pred_y)

        y_list_pred = np.concatenate(y_list_pred,axis=0)
        y_list = y_list.squeeze()
        y_list_pred = y_list_pred.squeeze()
        assert y_list_pred.shape[0] == x_list.shape[0]

        if not isinstance(y_list, np.ndarray):
            y_list = y_list.numpy()
        
        ###### calculate metrics ######

        mean_p = y_list_pred.mean(axis = 0)
        sigma_p = y_list_pred.std(axis = 0)
        mean_g = y_list.mean(axis = 0)
        sigma_g = y_list.std(axis = 0)

        index = (sigma_g!=0)
        corr = ((y_list_pred - mean_p) * (y_list - mean_g)).mean(axis = 0) / (sigma_p * sigma_g)
        corr = (corr[index]).mean()

        mse = (np.square(y_list_pred  - y_list )).mean()
        result_dict = {'mse':mse, 'r':corr, 'r^2':corr**2, 'rmse':np.sqrt(mse)}

        not_zero_idx = y_list != 0.0
        mape = (np.fabs(y_list_pred[not_zero_idx] -  y_list[not_zero_idx]) / np.fabs(y_list[not_zero_idx])).mean() * 100
        result_dict['mape'] = mape
        
    ### verbose ###
    if need_verbose:
        epoch_use_time = time.time() - epoch_start_time
        # valid -> interval time; final test -> all time
        print(name + 'corr = {:.4f}, rmse = {:.4f}, mape = {:.4f} %'.format(corr,np.sqrt(mse),mape) + ', time = {:.4f} s'.format(epoch_use_time))
        
    return result_dict


def save_reps_and_labels(args, model, x_list, y_list, device):

    # Create a copy of the model, and swap out the classification
    # layer for an identity.
    #print(model)
    model_copy = copy.deepcopy(model)
    model_copy.fclayer = torch.nn.Identity() 
    model_copy.eval()
    if args.dataset in ['CommunitiesAndCrime', 'SkillCraft']:
        model_copy.fclayer = torch.nn.Identity() 
        model_copy.eval()
        #raise ValueError('Dataset {} is not one of the tabular datasets'.format(args.dataset))
    elif args.dataset in ['RCF_MNIST', "ChairAngle_Tails"]:
        model_copy.fc = torch.nn.Identity() 
        model_copy.eval()        
    with torch.no_grad():
        val_len = x_list.shape[0]
        rep_list_pred = []

        if isinstance(x_list,np.ndarray):
            x_list_torch = torch.tensor(x_list, dtype=torch.float32).to(device)
        else:
            x_list_torch = x_list.to(device)

        if args.dataset == "ChairAngle_Tails":
            model_copy = model_copy.to(device)
            half_index = int(np.ceil(x_list.shape[0]/2))
            rep = model_copy(x_list_torch[:half_index]).cpu().numpy()
            rep_list_pred.append(rep)
            rep = model_copy(x_list_torch[half_index:]).cpu().numpy()
            rep_list_pred.append(rep)
        else:
            model_copy = model_copy.to(device)
            rep = model_copy(x_list_torch).cpu().numpy()
            rep_list_pred.append(rep)            

        rep_list_pred = np.concatenate(rep_list_pred,axis=0)
        y_list = y_list.squeeze()
        rep_list_pred = rep_list_pred.squeeze()

        if not isinstance(y_list, np.ndarray):
            y_list = y_list.numpy()
        
        print("Representation shapes!")
        print(rep_list_pred.shape)
        print(y_list.shape)
    
    return rep_list_pred, y_list



def train(args, model, data_packet, is_mixup=True, mixup_idx_sample_rate=None, ts_data= None, device='cuda'):
    ######### model prepare ########
    model.train(True)
    optimizer = Adam(model.parameters(), **args.optimiser_args)
    loss_fun = nn.MSELoss(reduction='mean').to(device)
    
    best_mse = 1e10  # for best update
    best_r2 = 0.0
    repr_flag = 1 # for batch kde visualize training process

    scheduler = None

    x_train = data_packet['x_train']
    y_train = data_packet['y_train']
    x_valid = data_packet['x_valid']
    y_valid = data_packet['y_valid']

    iteration = len(x_train) // args.batch_size
    steps_per_epoch = iteration

    result_dict,best_mse_model = {},None
    step_print_num = 30 # for dti

    need_shuffle = not args.is_ood

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        shuffle_idx = np.random.permutation(np.arange(len(x_train)))

        if need_shuffle: # id
            x_train_input = x_train[shuffle_idx]
            y_train_input = y_train[shuffle_idx]
        else:# ood
            x_train_input = x_train
            y_train_input = y_train

        if not is_mixup:

            # iteration for each batch
            for idx in range(iteration):
                # select batch
                x_input_tmp = x_train_input[idx * args.batch_size:(idx + 1) * args.batch_size]
                y_input_tmp = y_train_input[idx * args.batch_size:(idx + 1) * args.batch_size]

                # -> tensor
                if isinstance(x_input_tmp,np.ndarray):
                    x_input = torch.tensor(x_input_tmp, dtype=torch.float32).to(device)
                else:
                    x_input = x_input_tmp.to(device)

                if isinstance(y_input_tmp,np.ndarray):
                    y_input = torch.tensor(y_input_tmp, dtype=torch.float32).to(device)
                else:
                    y_input = y_input_tmp.to(device)

                # forward
                pred_Y = model(x_input)
                loss = loss_fun(pred_Y, y_input)


                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler != None: # backward (without scheduler)
                    scheduler.step()
                
                # validation
                if args.dataset == 'Dti_dg' and (idx-1)%(iteration//step_print_num)==0:
                    result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) + ', step = {} '.format((epoch*steps_per_epoch + idx))+':\t', args.show_process, epoch_start_time, device)
                    
                    # save best model
                    if result_dict['mse'] <= best_mse:
                        best_mse = result_dict['mse']
                        best_mse_model = copy.deepcopy(model)
                    if result_dict['r']**2 >= best_r2:
                        best_r2 = result_dict['r']**2
                        best_r2_model = copy.deepcopy(model)

        else:  # mix up
            for idx in range(iteration):

                lambd = np.random.beta(args.mix_alpha, args.mix_alpha)

                if need_shuffle: # get batch idx
                    idx_1 = shuffle_idx[idx * args.batch_size:(idx + 1) * args.batch_size]
                else:
                    idx_1 = np.arange(len(x_train))[idx * args.batch_size:(idx + 1) * args.batch_size]
                
                if args.mixtype == 'kde': 
                    idx_2 = np.array(
                        [np.random.choice(np.arange(x_train.shape[0]), p=mixup_idx_sample_rate[sel_idx]) for sel_idx in
                        idx_1])
                else: # random mix
                    idx_2 = np.array(
                        [np.random.choice(np.arange(x_train.shape[0])) for sel_idx in idx_1])

                if isinstance(x_train,np.ndarray):
                    X1 = torch.tensor(x_train[idx_1], dtype=torch.float32).to(device)
                    Y1 = torch.tensor(y_train[idx_1], dtype=torch.float32).to(device)

                    X2 = torch.tensor(x_train[idx_2], dtype=torch.float32).to(device)
                    Y2 = torch.tensor(y_train[idx_2], dtype=torch.float32).to(device)
                else:
                    X1 = x_train[idx_1].to(device)
                    Y1 = y_train[idx_1].to(device)

                    X2 = x_train[idx_2].to(device)
                    Y2 = y_train[idx_2].to(device)

                if args.batch_type == 1: # sample from batch
                    assert args.mixtype == 'random'
                    if not repr_flag: # show the sample status once
                        args.show_process = 0
                    else:
                        repr_flag = 0
                    X2, Y2 = get_batch_kde_mixup_batch(args,X1,X2,Y1,Y2,device)
                    args.show_process = 1

                X1 = X1.to(device)
                X2 = X2.to(device)
                Y1 = Y1.to(device)
                Y2 = Y2.to(device)

                # mixup
                mixup_Y = Y1 * lambd + Y2 * (1 - lambd)
                mixup_X = X1 * lambd + X2 * (1 - lambd)
                
                # forward
                if args.use_manifold == True:
                    pred_Y = model.forward_mixup(X1, X2, lambd)
                else:
                    pred_Y = model.forward(mixup_X)

                if args.dataset == 'TimeSeires': # time series loss need scale
                    scale = ts_data.scale.expand(pred_Y.size(0),ts_data.m)
                    loss = loss_fun(pred_Y * scale, mixup_Y * scale)
                else:    
                    loss = loss_fun(pred_Y, mixup_Y)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.dataset == 'Dti_dg' and (idx-1) % (iteration // step_print_num) == 0: # dti has small epoch number, so verbose multiple times at 1 iteration
                    result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) + ',  step = {} '.format((epoch*steps_per_epoch + idx)) + ':\t', args.show_process, epoch_start_time, device)
                    # save best model
                    if result_dict['mse'] <= best_mse:
                        best_mse = result_dict['mse']
                        best_mse_model = copy.deepcopy(model)
                    if result_dict['r']**2 >= best_r2:
                        best_r2 = result_dict['r']**2
                        best_r2_model = copy.deepcopy(model)

        # validation
        result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) +':\t', args.show_process, epoch_start_time, device)
        


        if result_dict['mse'] <= best_mse:
            best_mse = result_dict['mse']
            best_mse_model = copy.deepcopy(model)
            print(f'update best mse! epoch = {epoch}')
        
        if result_dict['r']**2 >= best_r2:
            best_r2 = result_dict['r']**2
            best_r2_model = copy.deepcopy(model)

    return best_mse_model, best_r2_model


def coral_penalty(x, y):
    # From the following:
    # https://github.com/p-lambda/wilds/blob/main/examples/algorithms/deepCORAL.py#L55
    if x.dim() > 2:
        # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
        # we flatten to Tensors of size (*, feature dimensionality)
        x = x.view(-1, x.size(-1))
        y = y.view(-1, y.size(-1))

    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

    # NOTE: This line is included in the WILDS and Domainbed implementation of Deep CORAL.
    # It is not necessary for CORAL, but penalizing both first and second moment differences
    # seems potentially useful, so we keep it for consistency's sake.
    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff


def train_deep_coral(args, model, data_packet, mixup_idx_sample_rate=None, ts_data= None, device='cuda',
              coral_penalty_weight=1.0, full_batch=False):
    ######### model prepare ########
    model.train(True)
    optimizer = Adam(model.parameters(), **args.optimiser_args)
    loss_fun = nn.MSELoss(reduction='mean').to(device)
    
    best_mse = 1e10  # for best update
    best_r2 = 0.0
    repr_flag = 1 # for batch kde visualize training process

    scheduler = None

    x_train = data_packet['x_train']
    y_train = data_packet['y_train']
    x_valid = data_packet['x_valid']
    y_valid = data_packet['y_valid']
    x_test = data_packet['x_test']

    bsize = args.batch_size if not full_batch else len(x_train)

    if args.dataset in ['ChairAngle_Tails', 'RCF_MNIST']:
        if args.dataset == "RCF_MNIST":
            test_data_loader = DataLoader(TensorDataset(x_test),
                                batch_size=bsize, shuffle=True)
        else:
            test_data_loader = DataLoader(TensorDataset(torch.tensor(x_test, dtype=torch.float32)),
                                        batch_size=bsize, shuffle=True)
        test_data_batches = iter(test_data_loader)


    iteration = len(x_train) // bsize
    steps_per_epoch = iteration

    result_dict,best_mse_model = {},None
    step_print_num = 30 # for dti

    need_shuffle = not args.is_ood
    all_penalties = []
    for epoch in range(args.num_epochs):
        epoch_penalties = []
        epoch_start_time = time.time()
        model.train()
        shuffle_idx = np.random.permutation(np.arange(len(x_train)))

        if need_shuffle: # id
            x_train_input = x_train[shuffle_idx]
            y_train_input = y_train[shuffle_idx]

        else:# ood
            x_train_input = x_train
            y_train_input = y_train


        # iteration for each batch
        for idx in range(iteration):
            # select batch
            x_input_tmp = x_train_input[idx * bsize:(idx + 1) * bsize]
            y_input_tmp = y_train_input[idx * bsize:(idx + 1) * bsize]

            # -> tensor
            if args.dataset == "RCF_MNIST":
                x_input = x_input_tmp.to(device)
                y_input = y_input_tmp.to(device)
            else:
                x_input = torch.tensor(x_input_tmp, dtype=torch.float32).to(device)
                y_input = torch.tensor(y_input_tmp, dtype=torch.float32).to(device)

            # forward
            pred_Y, representation_x_input = model(x_input, return_reps=True)

            # First, sort the predictions and labels into domain based minibatches
            # Here, domains are simply train and test.
            # Add the test batch
            if args.dataset in ['ChairAngle_Tails', 'RCF_MNIST']:
                try:
                    # get a batch of unlabeled data from target domain
                    x_test_batch = next(test_data_batches)
                    x_test_batch_data = x_test_batch[0].to(device)
                    x_test_input = x_test_batch_data[:x_input.shape[0]]
                    assert x_test_input.shape[0] == x_input.shape[0], 'Target batch was small while source batch was big'
                except StopIteration: # ran out of UDA data
                    test_data_batches = iter(test_data_loader)
                    x_test_batch = next(test_data_batches)
                    x_test_batch_data = x_test_batch[0].to(device)
                    x_test_input = x_test_batch_data[:x_input.shape[0]]
                    assert x_test_input.shape[0] == x_input.shape[0], 'Target batch was small while source batch was big'
                except AssertionError: # this UDA batch isn't full sized; skip it
                    test_data_batches = iter(test_data_loader)
                    x_test_batch = next(test_data_batches)
                    x_test_batch_data = x_test_batch[0].to(device)
                    x_test_input = x_test_batch_data[:x_input.shape[0]]
                    assert x_test_input.shape[0] == x_input.shape[0], 'Target batch was small while source batch was big'
            else:
                x_test_input = torch.tensor(x_test, dtype=torch.float32).to(device)

            throwaway, representation_x_test_input = model(x_test_input, return_reps=True)

            del throwaway

            batch_penalty = coral_penalty(representation_x_input, representation_x_test_input)
            epoch_penalties.append(batch_penalty.detach().cpu().item())

            # Scale the penalty
            scaled_batch_penalty = coral_penalty_weight*batch_penalty.squeeze()

            # Calculate the main loss
            loss = loss_fun(pred_Y, y_input)
            print("Loss is {}".format(loss))
            loss += scaled_batch_penalty
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None: # backward (without scheduler)
                scheduler.step()
            
            # validation
            if args.dataset == 'Dti_dg' and (idx-1)%(iteration//step_print_num)==0:
                result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) + ', step = {} '.format((epoch*steps_per_epoch + idx))+':\t', args.show_process, epoch_start_time, device)
                
                # save best model
                if result_dict['mse'] <= best_mse:
                    best_mse = result_dict['mse']
                    best_mse_model = copy.deepcopy(model)
                if result_dict['r']**2 >= best_r2:
                    best_r2 = result_dict['r']**2
                    best_r2_model = copy.deepcopy(model)

        all_penalties.append(np.mean(epoch_penalties))
        # validation
        result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) +':\t', args.show_process, epoch_start_time, device)
        


        if result_dict['mse'] <= best_mse:
            best_mse = result_dict['mse']
            best_mse_model = copy.deepcopy(model)
            print(f'update best mse! epoch = {epoch}')
        
        if result_dict['r']**2 >= best_r2:
            best_r2 = result_dict['r']**2
            best_r2_model = copy.deepcopy(model)

    print("All the penalties!")
    print(all_penalties)
    plt.plot(np.linspace(0, len(all_penalties), len(all_penalties)), all_penalties)
    plt.xlabel("Epoch")
    plt.ylabel("Average Coral Penalty across epochs")
    plt.title("Lambda = {}, bsize = {}".format(coral_penalty_weight, bsize))
    plt.savefig("./coral_penalties_lam_{}_bsize_{}.png".format(coral_penalty_weight, bsize))
    return best_mse_model, best_r2_model


class Discriminator(nn.Module):
    def __init__(self, args, hid_dim = 128):
        super(Discriminator, self).__init__()
        self.block_1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.LeakyReLU(0.1))
        self.fclayer = nn.Sequential(nn.Linear(hid_dim, 2))

    def forward(self, x):
        x = self.block_1(x)
        output = self.fclayer(x)
        return output


def train_dann(args, model, data_packet, mixup_idx_sample_rate=None, ts_data= None, device='cuda',
              disc_penalty_weight=1.0, full_batch=False):
    ######### model prepare ########
    model.train(True)
    optimizer = Adam(model.parameters(), **args.optimiser_args)
    loss_fun = nn.MSELoss(reduction='mean').to(device)

    # Set up the discriminator and its optimizer.
    if args.dataset in ["ChairAngle_Tails", "RCF_MNIST"]:
        discriminator = Discriminator(args=args, hid_dim=512).to(device)
    else:
        discriminator = Discriminator(args=args, hid_dim=128).to(device)
    disc_optimizer = Adam(discriminator.parameters(), lr=args.disc_lr)
    disc_loss_function = nn.CrossEntropyLoss(reduction='mean').to(device)
    discriminator.train()
    
    best_mse = 1e10  # for best update
    best_r2 = 0.0
    repr_flag = 1 # for batch kde visualize training process

    scheduler = None

    x_train = data_packet['x_train']
    y_train = data_packet['y_train']
    x_valid = data_packet['x_valid']
    y_valid = data_packet['y_valid']
    x_test = data_packet['x_test']

    bsize = args.batch_size if not full_batch else len(x_train)

    if args.dataset in ['ChairAngle_Tails', 'RCF_MNIST']:
        if args.dataset == "RCF_MNIST":
            test_data_loader = DataLoader(TensorDataset(x_test),
                                batch_size=bsize, shuffle=True)
        else:
            test_data_loader = DataLoader(TensorDataset(torch.tensor(x_test, dtype=torch.float32)),
                                        batch_size=bsize, shuffle=True)
        test_data_batches = iter(test_data_loader)


    iteration = int(np.ceil(len(x_train) / bsize))
    steps_per_epoch = iteration

    result_dict,best_mse_model = {},None
    step_print_num = 30 # for dti

    need_shuffle = not args.is_ood
    all_penalties = []
    for epoch in range(args.num_epochs):
        epoch_penalties = []
        epoch_start_time = time.time()
        model.train()
        discriminator.train()
        shuffle_idx = np.random.permutation(np.arange(len(x_train)))

        if need_shuffle: # id
            x_train_input = x_train[shuffle_idx]
            y_train_input = y_train[shuffle_idx]
        else:# ood
            x_train_input = x_train
            y_train_input = y_train


        # iteration for each batch
        for idx in range(iteration):
            # select batch
            x_input_tmp = x_train_input[idx * bsize:(idx + 1) * bsize]
            y_input_tmp = y_train_input[idx * bsize:(idx + 1) * bsize]

            # -> tensor
            if args.dataset == "RCF_MNIST":
                x_input = x_input_tmp.to(device)
                y_input = y_input_tmp.to(device)
            else:
                x_input = torch.tensor(x_input_tmp, dtype=torch.float32).to(device)
                y_input = torch.tensor(y_input_tmp, dtype=torch.float32).to(device)

            # forward
            pred_Y, representation_x_input = model(x_input, return_reps=True)

            # Get the test representations as well
            # Add the test batch
            if args.dataset in ['ChairAngle_Tails', 'RCF_MNIST']:
                try:
                    # get a batch of unlabeled data from target domain
                    x_test_batch = next(test_data_batches)
                    x_test_batch_data = x_test_batch[0].to(device)
                    x_test_input = x_test_batch_data[:x_input.shape[0]]
                    assert x_test_input.shape[0] == x_input.shape[0], 'Target batch was small while source batch was big'
                except StopIteration: # ran out of UDA data
                    test_data_batches = iter(test_data_loader)
                    x_test_batch = next(test_data_batches)
                    x_test_batch_data = x_test_batch[0].to(device)
                    x_test_input = x_test_batch_data[:x_input.shape[0]]
                    assert x_test_input.shape[0] == x_input.shape[0], 'Target batch was small while source batch was big'
                except AssertionError: # this UDA batch isn't full sized; skip it
                    test_data_batches = iter(test_data_loader)
                    x_test_batch = next(test_data_batches)
                    x_test_batch_data = x_test_batch[0].to(device)
                    x_test_input = x_test_batch_data[:x_input.shape[0]]
                    assert x_test_input.shape[0] == x_input.shape[0], 'Target batch was small while source batch was big'
            else:
                x_test_input = torch.tensor(x_test, dtype=torch.float32).to(device)
            throwaway, representation_x_test = model(x_test_input, return_reps=True)
            del throwaway

            # Create "domain labels" (train or test) for the discriminator loss
            train_domain_labels = torch.zeros(representation_x_input.shape[0])
            test_domain_labels = torch.ones(representation_x_test.shape[0])

            all_domain_labels = torch.cat([train_domain_labels, test_domain_labels], axis=0).to(device).long()
            all_representations = torch.cat([representation_x_input, representation_x_test], axis=0)

            # Get the discriminator's loss
            disc_preds = discriminator(all_representations)


            disc_loss = disc_loss_function(disc_preds, all_domain_labels)

            epoch_penalties.append(disc_loss.detach().cpu().item())

            # Scale the penalty
            # NOTE: We scale it by negative one times the penalty weight
            # so that we maximize the discrimnator loss.
            scaled_batch_penalty = -disc_penalty_weight*disc_loss.squeeze()

            # Calculate the main loss
            loss = loss_fun(pred_Y, y_input)
            print("Loss is {}".format(loss))
            loss += scaled_batch_penalty
            # backward
            disc_optimizer.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None: # backward (without scheduler)
                scheduler.step()
            
            # validation
            if args.dataset == 'Dti_dg' and (idx-1)%(iteration//step_print_num)==0:
                result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) + ', step = {} '.format((epoch*steps_per_epoch + idx))+':\t', args.show_process, epoch_start_time, device)
                
                # save best model
                if result_dict['mse'] <= best_mse:
                    best_mse = result_dict['mse']
                    best_mse_model = copy.deepcopy(model)
                if result_dict['r']**2 >= best_r2:
                    best_r2 = result_dict['r']**2
                    best_r2_model = copy.deepcopy(model)


        # Now train the discriminator
        # Reset the test dataloader if required
        if args.dataset in ['ChairAngle_Tails', 'RCF_MNIST']:
            test_data_batches = iter(test_data_loader)
        # iteration for each batch
        for idx in range(iteration):
            # select batch
            x_train_input_tmp = x_train_input[idx * bsize:(idx + 1) * bsize]

            # -> tensor
            if args.dataset == "RCF_MNIST":
                x_train_input_tensor = x_train_input_tmp.to(device)
            else:
                x_train_input_tensor = torch.tensor(x_train_input_tmp, dtype=torch.float32).to(device)

            # Get representations for both
            # forward
            throwaway, representation_x_train_input_all = model(x_train_input_tensor, return_reps=True)

            del throwaway

            # Get the test representations as well
            # Add the test batch
            if args.dataset in ['ChairAngle_Tails', 'RCF_MNIST']:
                try:
                    # get a batch of unlabeled data from target domain
                    x_test_batch = next(test_data_batches)
                    x_test_batch_data = x_test_batch[0].to(device)
                    x_test_input = x_test_batch_data[:x_train_input_tensor.shape[0]]
                    assert x_test_input.shape[0] == x_train_input_tensor.shape[0], 'Target batch was small while source batch was big'
                except StopIteration: # ran out of UDA data
                    test_data_batches = iter(test_data_loader)
                    x_test_batch = next(test_data_batches)
                    x_test_batch_data = x_test_batch[0].to(device)
                    x_test_input = x_test_batch_data[:x_train_input_tensor.shape[0]]
                    assert x_test_input.shape[0] == x_train_input_tensor.shape[0], 'Target batch was small while source batch was big'
                except AssertionError: # this UDA batch isn't full sized; skip it
                    test_data_batches = iter(test_data_loader)
                    x_test_batch = next(test_data_batches)
                    x_test_batch_data = x_test_batch[0].to(device)
                    x_test_input = x_test_batch_data[:x_train_input_tensor.shape[0]]
                    assert x_test_input.shape[0] == x_train_input_tensor.shape[0], 'Target batch was small while source batch was big'
            else:
                x_test_input = torch.tensor(x_test, dtype=torch.float32).to(device)

            # Get the test representations as well
            # NOTE: For now, we always include ALL test examples.
            throwaway, representation_x_test_all = model(x_test_input, return_reps=True)
            del throwaway

            # Create "domain labels" (train or test) for the discriminator loss
            all_train_domain_labels = torch.zeros(representation_x_train_input_all.shape[0])
            all_test_domain_labels = torch.ones(representation_x_test_all.shape[0])

            all_domain_labels = torch.cat([all_train_domain_labels, all_test_domain_labels], axis=0).to(device).long()
            all_representations = torch.cat([representation_x_train_input_all, representation_x_test_all], axis=0)

            assert all_domain_labels.shape[0] == all_representations.shape[0]
            assert all_representations.shape[0] == (x_test_input.shape[0] + x_train_input_tensor.shape[0])

            # Get the discriminator's loss
            disc_preds = discriminator(all_representations)

            disc_loss = disc_loss_function(disc_preds, all_domain_labels)

            print("The total discriminator loss was: {}".format(disc_loss))

            # Optimize the discriminator
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

        all_penalties.append(np.mean(epoch_penalties))
        # validation
        result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) +':\t', args.show_process, epoch_start_time, device)
        


        if result_dict['mse'] <= best_mse:
            best_mse = result_dict['mse']
            best_mse_model = copy.deepcopy(model)
            print(f'update best mse! epoch = {epoch}')
        
        if result_dict['r']**2 >= best_r2:
            best_r2 = result_dict['r']**2
            best_r2_model = copy.deepcopy(model)

    print("All the penalties!")
    print(all_penalties)
    plt.plot(np.linspace(0, len(all_penalties), len(all_penalties)), all_penalties)
    plt.xlabel("Epoch")
    plt.ylabel("Average Discriminator loss across epochs")
    plt.title("Lambda = {}, bsize = {}".format(disc_penalty_weight, bsize))
    plt.savefig("./DANN_penalties_lam_{}_bsize_{}.png".format(disc_penalty_weight, bsize))
    return best_mse_model, best_r2_model

