#!/bin/bash

python3 -u train_megan.py \
-g_layers   128 64 1 \
-g_num      4 \
-d_layers   64 128 256 \
-d_norm     "layer_norm" \
-input_shape 256 4 4 \
-z_dim      100 \
-batch_size 128 \
-test_size  10000 \
-lr         1e-4 \
-lr_decay   1.0 \
-lr_step    1 \
-epoch      50 \
-out        "out/MEGAN/x" \
-seed       1 \
-device     "cuda:0" \
-dataset    "mnist" \
-c_iter     5 \
-topk       5 \
-test_step  5 \
-img_step   1 \
-acc        0 

