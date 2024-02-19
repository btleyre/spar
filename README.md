# PovertyMap experiments
This subdirectory supplements the PovertyMap experiments in our paper [Out of the Ordinary: Spectrally Adapting Regression for Covariate Shift](https://arxiv.org/abs/2312.17463).

base_commands.sh contains commands for reproducing results on a single seed for ERM and C-Mixup. Note that main.py must be run twice for each experiment: once to produce the encoder, and once to produce the projected regressor (using the --project flag). Experiments were run on 5 seeds, seeds 0-4.

This code is adapted from [the src folder of the C-Mixup codebase](https://github.com/huaxiuyao/C-Mixup/tree/main/src), which is subject to the MIT license.