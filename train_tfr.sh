#!/usr/bin/env bash
#<<comment
python trains/trainer_tfr.py \
--train_list_path=/Users/raymond/Desktop/git_local/lwlearntools/data_create/train0.tfr \
--test_list_path=/Users/raymond/Desktop/git_local/lwlearntools/data_create/test0.tfr \
--ckpt_dir=/Users/raymond/Desktop/git_local/lwlearntools/data_create \
--batch_size=64 \
--gpu_id=0,1 \
--model_type=desnetSeq \
--class_type=addr \
--disp_inter=100 \
--test_inter=1000 \
--save_inter=1000 \
--show_detail=True \
--mode_flag=train \
--lr_type=sgdr \
--need_padded_batch=0 \
--opt_type=sgd \
--show_detail_summary=1 \
--total_epochs=100

#comment