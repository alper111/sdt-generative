#!/bin/bash

python3 -u train_madgan.py \
-g_model    "conv" \
-g_layers   16 8 1 \
-g_num      5 \
-g_norm     "batch_norm" \
-d_model    "conv" \
-d_layers   8 16 32 \
-d_norm     "batch_norm" \
-input_shape 32 4 4 \
-z_dim      100 \
-batch_size 128 \
-test_size  10000 \
-lr         1e-4 \
-lr_decay   1.0 \
-lr_step    1 \
-wasserstein 0 \
-epoch      50 \
-out        "out/MADGAN/test" \
-seed       1 \
-device     "cuda:0" \
-dataset    "mnist" \
-c_iter     1 \
-topk       5 \
-test_step  5 \
-img_step   1 \
-acc        0
