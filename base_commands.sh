
# Run the seed 0 base experiment for ERM on Povertymap
time python3 main.py --dataset poverty --algorithm erm --data-dir ../../datasets/ --experiment_dir .. --is_kde 0 --kde_bandwidth 0.5 --mix_alpha 0.5 --seed 0 --spar_alpha 0.999 --search_lr 0.001 --search_artifact_base_path ./search_results/base_erm_seed_0

# Perform projection using the ERM model. NOTE: only run this command after the previous command has terminated.
# Replace $1 with the directory containing the trained ERM model.rar file. 

time python3 main.py --dataset poverty --algorithm erm --data-dir ../../datasets/ --experiment_dir .. --is_kde 0 --projection --base_model_path $1 --seed 0 --proj_artifact_dir ./projection_results/base_erm_seed_0 --kde_bandwidth 0.5 --search_lr 0.001 --spar_alpha 0.999 --adapt_to_unlabeled_test_data

# Run the seed 0 base experiment for C-Mixup on Povertymap
time python3 main.py --dataset poverty --algorithm mixup --data-dir ../../datasets/ --experiment_dir .. --is_kde 1 --kde_bandwidth 0.5 --mix_alpha 0.5 --seed 0 --spar_alpha 0.999 --search_lr 0.001 --search_artifact_base_path ./search_results/base_cmixup_seed_0

# Perform projection using the C-mixup model. NOTE: only run this command after the previous command has terminated.
# Replace $2 with the directory containing the trained C-mixup model.rar file. 

time python3 main.py --dataset poverty --algorithm mixup --data-dir ../../datasets/ --experiment_dir .. --is_kde 1 --projection --base_model_path $2 --seed 0 --proj_artifact_dir ./projection_results/base_cmixup_seed_0 --kde_bandwidth 0.5 --search_lr 0.001 --spar_alpha 0.999 --adapt_to_unlabeled_test_data

