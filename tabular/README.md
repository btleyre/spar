# Tabular experiments
This subdirectory supplements the tabular experiments in the SpAR paper.

base_commands.sh contains 4 commands that can be used for running experiments with C-mixup/ERM on CommunitiesAndCrime/Skillcraft. Experiments were run on ten seeds, seeds 0-9.

This code is adapted from [the src folder of the C-Mixup codebase](https://github.com/huaxiuyao/C-Mixup/tree/main/src), which is subject to the MIT license.
The code within the `data/Dti_dg_lib` folder from the C-Mixup codebase follows a different lisence.
Additionally, the `/tabular/net/` folder is sourced from the [repo for the paper "How Reliable is Your Regression Model's Uncertainty Under Real-World Distribution Shifts?"](https://github.com/fregu856/regression_uncertainty).

# CommunitiesAndCrime, SkillCraft, and RCF-MNIST Datasets

Before running experiments on these datasets, follow the data download instructions from the C-Mixup codebase.

[CommunitiesAndCrime](https://github.com/huaxiuyao/C-Mixup/tree/main?tab=readme-ov-file#crime)

[SkillCraft](https://github.com/huaxiuyao/C-Mixup/tree/main?tab=readme-ov-file#skillcraft)

[RCF-MNIST](https://github.com/huaxiuyao/C-Mixup/tree/main?tab=readme-ov-file#rcf-mnist)

# ChairAngles-Tails Dataset

Before running experiments from the ChairAngles-Tails dataset, be sure to first produce the `.pkl` files using the instructions in this repo: https://github.com/fregu856/regression_uncertainty/tree/main?tab=readme-ov-file#chairangle-tails

Once that is done, place the pickle files in a new subfolder `./tabular/data/ChairAngle_Tails`.

# Dependencies

Python 3.7.13 was used. Package versions are included in `src_requirements.txt`.

Additionally, `torch-scatter` must be installed using the following command.

`pip3 install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu102.html`