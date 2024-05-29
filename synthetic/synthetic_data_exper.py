"""Script for conducting the synthetic experiments with SpAR."""
import argparse
import copy
import random
import numpy as np
from numpy.random import multivariate_normal
import torch
from torch.nn import MSELoss
import scipy
from scipy.stats import chi2
from sklearn.decomposition import PCA


GLOBAL_DIM = 2
GLOBAL_NUM_SAMPLES = 4000
GLOBAL_NOISE_RATE = 1
GLOBAL_SEED = 123


def compute_pseudoinvese_soln(X_train_representations, Y_train_labels):
    """Return the pinv soln, X^+Y."""
    train_pinv_soln = np.matmul(
        scipy.linalg.pinv(X_train_representations),
        Y_train_labels
    ).squeeze() 
    return train_pinv_soln[:, None]


def spar_chi_adaptation(args, X_train_representations, Z_test_representations, Y_train_labels, sigma_squared):
    """Adapt the OLS regressor according to our SpAR-Chi approach.

    Parameters:
        args: parsed command line arguments.
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

    print("Removing these singular vectors: {}".format(chi2_remove_map))

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

        chi2_remove_vector = np.sum(
            chi2_bad_eigvec_projection_weights*chi2_eigvecs_to_be_removed,
            axis=0
        ).squeeze()
        chi2_w_proj = train_pinv_soln - chi2_remove_vector

    # Structure the output as a Dx1 vector.
    assert len(chi2_w_proj.shape) == 1
    return chi2_w_proj[:, None]


def top_k_spar_adaptation(args, X_train_representations, Z_test_representations, Y_train_labels, select_k):
    """Adapt the OLS regressor using uncentred PCR.

    Parameters:
        args: parsed command line arguments.
        X_train_representations: torch tensor. The training representations.
        Z_test_representations: torch tensor. The test representations.
        Y_train_labels: torch tensor. The training labels.
        select_k: int. The number of top OOD eigenvectors to keep.

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
    null_vh_x = copy.deepcopy(vh_x)[train_rank:]
    vh_x = vh_x[:train_rank]

    # Calculate the pseudoinverse solution, and use it to estimate the bias term
    train_pinv_soln = compute_pseudoinvese_soln(X_train_representations, Y_train_labels).squeeze()

    # Take only the top k eigenvectors
    remove_map = np.arange(X_train_representations.shape[1])
    remove_map = remove_map >= select_k

    num_evecs_retained = np.sum((~(remove_map)).astype(int))

    eigvecs_to_be_removed = vh_z[remove_map, :]

    if eigvecs_to_be_removed.shape[0] == 0:
        # In this case, no evecs to be removed, so we just keep it as ols
        w_proj = train_pinv_soln
    else:

        if eigvecs_to_be_removed.shape[0] == 1:
            bad_eigvec_projection_weights = np.matmul(eigvecs_to_be_removed, train_pinv_soln)[:, None]
        else:
            bad_eigvec_projection_weights = np.matmul(eigvecs_to_be_removed, train_pinv_soln).squeeze()[:, None]

        remove_vector = np.sum(
            bad_eigvec_projection_weights*eigvecs_to_be_removed,
            axis=0
        ).squeeze()
        w_proj = train_pinv_soln - remove_vector
    # Structure the output as a Dx1 vector.
    assert len(w_proj.shape) == 1
    return w_proj[:, None]


def main():
    parser = argparse.ArgumentParser(description='OLS Synthetic Experiment')
    parser.add_argument('--num_repeats', type=int)
    parser.add_argument('--regressor_choice', type=int, default=1)
    parser.add_argument('--spar_alpha', type=float, default=0.999)
    parser.add_argument("--high_dim", default=False, action='store_true')


    flags = parser.parse_args()

    ols_mses = []
    spar_mses = []
    mle_estimates = []
    first_pc_ols_mses = []
    no_center_first_pc_test_preds_mses = []

    for repeat_int in range(int(flags.num_repeats)):


        print('Flags:')
        for k,v in sorted(vars(flags).items()):
            print("\t{}: {}".format(k, v))

        # Fix the randomness
        torch.manual_seed(GLOBAL_SEED + repeat_int)
        np.random.seed(GLOBAL_SEED + repeat_int)
        random.seed(GLOBAL_SEED + repeat_int)

        if not flags.high_dim:

            # Create a "True" labelling vector
            if flags.regressor_choice == 1:
                true_vector = np.array([0,1])*10 + np.array([1,0])*1e-2
            elif flags.regressor_choice == 2:
                true_vector = np.array([1,0])*10 + np.array([0,1])*1e-2
            elif flags.regressor_choice == 3 or flags.regressor_choice == 4:
                true_vector = np.array([0,1])*2 + np.array([1,0])*1
            else:
                raise ValueError("{} is not a valid regressor choice".format(flags.regressor_choice))

            true_vector = true_vector/np.linalg.norm(true_vector.squeeze())

            print("True vector is")
            print(true_vector)

            # Sample a bunch of multivariate gaussian vectors.
            old_covariance = np.identity(GLOBAL_DIM)

            if flags.regressor_choice != 4:
                old_covariance[1,1] = 1e-5
                old_covariance[0,0] = 5
            else:
                # Match the test covariance
                old_covariance[1,1] = 40


            X_train = multivariate_normal(np.zeros(GLOBAL_DIM), old_covariance,  size=(GLOBAL_NUM_SAMPLES))

            # Generate the labels (matrix multiplication + adding noise)
            noise = np.random.normal(scale=GLOBAL_NOISE_RATE, size=GLOBAL_NUM_SAMPLES)
            Y_train_base = np.matmul(X_train, true_vector) 
            Y_train = Y_train_base + noise

            # Generate an OOD set and the corresponding labels
            new_covariance = np.identity(GLOBAL_DIM)
            new_covariance[1,1] = 40

            Z_test = multivariate_normal(np.zeros(GLOBAL_DIM), new_covariance,  size=(GLOBAL_NUM_SAMPLES))
            Y_test_base = np.matmul(Z_test, true_vector)
            Y_test = Y_test_base
        else:
            # High dimensional synthetic data experiment.
            HIGH_DIM = 100
            # Randomly sample the labelling vector
            true_vector = multivariate_normal(np.zeros(HIGH_DIM), np.identity(HIGH_DIM),  size=(1)).squeeze()
            true_vector = true_vector/np.linalg.norm(true_vector.squeeze())

            # randomly sample two means
            id_mean = multivariate_normal(np.zeros(HIGH_DIM), np.identity(HIGH_DIM),  size=(1)).squeeze()
            ood_mean = multivariate_normal(np.zeros(HIGH_DIM), np.identity(HIGH_DIM),  size=(1)).squeeze()

            # Sample a random OOD covariance
            ood_covariance = np.diag(np.absolute(multivariate_normal(np.zeros(HIGH_DIM), np.identity(HIGH_DIM),  size=(1)).squeeze()))

            # Sample a random ID covariance, but scale the entries.
            id_covar_diagonal = np.absolute(multivariate_normal(np.zeros(HIGH_DIM), np.identity(HIGH_DIM),  size=(1)).squeeze())
            for block_num in range(10):
                id_covar_diagonal[block_num*10:(block_num+1)*10] = id_covar_diagonal[block_num*10:(block_num+1)*10]*(10**(7-block_num))
            id_covariance = np.diag(id_covar_diagonal)

            # Generate the data according to these random specifications
            X_train = multivariate_normal(id_mean, id_covariance,  size=(GLOBAL_NUM_SAMPLES))

            # Generate the labels (matrix multiplication + adding noise)
            noise = np.random.normal(scale=GLOBAL_NOISE_RATE, size=GLOBAL_NUM_SAMPLES)
            Y_train_base = np.matmul(X_train, true_vector) 
            Y_train = Y_train_base + noise

            Z_test = multivariate_normal(ood_mean, ood_covariance,  size=(GLOBAL_NUM_SAMPLES))
            Y_test_base = np.matmul(Z_test, true_vector)
            Y_test = Y_test_base

        # Calculate the OLS classifier
        ols = np.matmul(
            scipy.linalg.pinv(X_train),
            Y_train
        )

        train_preds = np.matmul(X_train, ols)
        train_mse_loss = MSELoss(reduction='sum')(
            torch.from_numpy(train_preds),
            torch.from_numpy(Y_train),
        )
        print("TRAIN MSE was {}".format(train_mse_loss))

        # Estimate the variance of the noise using the train mse
        MLE_sigma_squared_estimate = train_mse_loss/GLOBAL_NUM_SAMPLES

        print("MLE estimate for sigma^2 was {}".format(MLE_sigma_squared_estimate))
        mle_estimates.append(MLE_sigma_squared_estimate)

        test_preds = np.matmul(Z_test, ols)
        test_mse_loss = MSELoss(reduction='sum')(
            torch.from_numpy(test_preds),
            torch.from_numpy(Y_test),
        )
        print("OOD MSE was {}".format(test_mse_loss))
        ols_mses.append(test_mse_loss)


        # Benchmark our method, SpAR.

        spar_w_proj = spar_chi_adaptation(flags, X_train, Z_test, Y_train, MLE_sigma_squared_estimate.numpy()).squeeze()

        spar_proj_test_preds = np.matmul(Z_test, spar_w_proj)
        spar_test_mse_loss = MSELoss(reduction='sum')(
            torch.from_numpy(spar_proj_test_preds),
            torch.from_numpy(Y_test),
        )
        print("SpAR MSE was {}".format(spar_test_mse_loss))
        spar_mses.append(spar_test_mse_loss)

        # Benchmark principal component regression
        pca = PCA()
        X_train_pcs = pca.fit_transform(X_train)
        Z_test_pcs = pca.transform(Z_test)
        print("Principal Component X_train dim: {}".format(X_train_pcs.shape))

        # Next, benchmark the performance of rotation onto only the first component.
        first_component_X_train = X_train_pcs[:,0]
        first_component_X_train = first_component_X_train[:, None]
        first_component_Z_test = Z_test_pcs[:,0]
        first_component_Z_test = first_component_Z_test[:, None]
        print("First Principal Component X_train dim: {}".format(first_component_X_train.shape))
        first_pc_ols = np.matmul(
            scipy.linalg.pinv(first_component_X_train),
            Y_train
        )
        first_pc_test_preds = np.matmul(first_component_Z_test, first_pc_ols)
        first_pc_test_mse_loss = MSELoss(reduction='sum')(
            torch.from_numpy(first_pc_test_preds),
            torch.from_numpy(Y_test),
        )
        print("First PC OOD MSE was {}".format(first_pc_test_mse_loss))
        first_pc_ols_mses.append(first_pc_test_mse_loss)

        no_center_first_pc = top_k_spar_adaptation(flags, X_train, X_train, Y_train, 1).squeeze()

        no_center_first_pc_test_preds = np.matmul(Z_test, no_center_first_pc)
        no_center_first_pc_test_preds_test_mse_loss = MSELoss(reduction='sum')(
            torch.from_numpy(no_center_first_pc_test_preds),
            torch.from_numpy(Y_test),
        )
        print("no_center_first_pc MSE was {}".format(no_center_first_pc_test_preds_test_mse_loss))
        no_center_first_pc_test_preds_mses.append(no_center_first_pc_test_preds_test_mse_loss)


    print("Avg ols mse: {}+-{}".format(np.mean(ols_mses), np.std(ols_mses)))
    print("Avg SpAR mse: {}+-{}".format(np.mean(spar_mses), np.std(spar_mses)))
    print("Avg First PC OLS mse: {}+-{}".format(np.mean(first_pc_ols_mses), np.std(first_pc_ols_mses)))
    print("Avg No Center First PC OLS mse: {}+-{}".format(np.mean(no_center_first_pc_test_preds_mses), np.std(no_center_first_pc_test_preds_mses)))

    print("Avg mle estimate of sigma^2: {}+-{}".format(np.mean(mle_estimates), np.std(mle_estimates)))

if __name__ == "__main__":
    main()
