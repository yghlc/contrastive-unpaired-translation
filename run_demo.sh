#!/bin/bash

# donwload data
#bash ./datasets/download_cut_dataset.sh grumpifycat

# run training CUT model
#python train.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT


# test CUT model
python test.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT --phase train



# train FastCUT model
#python train.py --dataroot ./datasets/grumpifycat --name grumpycat_FastCUT --CUT_mode FastCUT

