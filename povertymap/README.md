# PovertyMap experiments
This subdirectory supplements the PovertyMap experiments in the SpAR paper.

base_commands.sh contains commands for reproducing results on a single seed for ERM and C-Mixup. Note that main.py must be run twice for each experiment: once to produce the encoder, and once to produce the projected regressor (using the --project flag). Experiments were run on 5 seeds, seeds 0-4.

This code is adapted from [the PovertyMap folder of the C-Mixup codebase](https://github.com/huaxiuyao/C-Mixup/tree/main/src), which is subject to the MIT license.

# Dependencies

Python 3.7.13 was used. Package versions are included in `povertymap_requirements.txt`.

Additionally, `torch-scatter` must be installed using the following command.

`pip3 install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu102.html`