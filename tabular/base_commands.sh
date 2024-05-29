
# Run the seed 0 base experiment for ERM on Skillcraft
python3 main.py --dataset SkillCraft --mixtype erm --kde_bandwidth 5e-4 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 0 --spar_alpha 0.999 --search_lr 0.01 --search_artifact_base_path ./search_results/SkillCraft/base_erm_seed_0

# Run the seed 0 base experiment for Cmixup on Skillcraft
python3 main.py --dataset SkillCraft --mixtype kde --kde_bandwidth 5e-4 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 0 --spar_alpha 0.999 --search_lr 0.01 --search_artifact_base_path ./search_results/SkillCraft/base_kde_seed_0

# Run the seed 0 base experiment for ERM on CommunitiesAndCrime
python3 main.py --dataset CommunitiesAndCrime --mixtype erm --kde_bandwidth 1.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0 --spar_alpha 0.999 --search_lr 0.001 --search_artifact_base_path ./search_results/CommunitiesAndCrime/base_erm_seed_0

# Run the seed 0 base experiment for Cmixup on CommunitiesAndCrime
python3 main.py --dataset CommunitiesAndCrime --mixtype kde --kde_bandwidth 1.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0 --spar_alpha 0.999 --search_lr 0.001 --search_artifact_base_path ./search_results/CommunitiesAndCrime/base_kde_seed_0

# Run the seed 0 tuned experiment for ERM/Cmixup on the RCF MMNIST dataset

python3 main.py --dataset RCF_MNIST --data_dir ./data/RCF_MNIST --batch_type 1 --mixtype erm --kde_bandwidth 0.2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0 --spar_alpha 0.999 --search_lr 3.851230830192189e-05 --search_artifact_base_path ./search_results/RCF_MNIST/base_erm_seed_0

python3 main.py --dataset RCF_MNIST --data_dir ./data/RCF_MNIST --batch_type 1 --mixtype random --kde_bandwidth 0.20091864782463165 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0 --spar_alpha 0.999 --search_lr 3.636910217027964e-05 --search_artifact_base_path ./search_results/RCF_MNIST/base_random_seed_0

# Run the seed 0 base tuned for ERM/Cmixup on the ChairAngles-Tails dataset.

python3 main.py --dataset ChairAngle_Tails --mixtype erm --kde_bandwidth 5e-4 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 0 --spar_alpha 0.999 --search_lr 0.00035721403189963735 --search_artifact_base_path ./search_results/ChairAngles_Tails/base_erm_seed_0

python3 main.py --dataset ChairAngle_Tails --mixtype kde --kde_bandwidth 0.0018607463328684177 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 0 --spar_alpha 0.999 --search_lr 0.0007506810936068544 --search_artifact_base_path ./search_results/ChairAngles_Tails/base_kde_seed_0
