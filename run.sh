#!/bin/bash

python3 train_vanilla_gan.py \
-g_model    "conv" \
-g_layers   128 64 1 \
-g_depth    5 \
-g_proj     "constant" \
-g_drop     0.0 \
-d_model    "conv" \
-d_layers   64 128 256 \
-d_depth    0 \
-d_proj     "constant" \
-d_norm     "layer_norm" \
-input_shape 256 4 4 \
-z_dim      100 \
-batch_size 128 \
-test_size  10000 \
-lr         1e-4 \
-lr_decay   1.0 \
-lr_step    1 \
-wasserstein 1 \
-epoch      250 \
-out        "out/deneme" \
-seed       2019 \
-device     "cpu" \
-dataset    "mnist" \
-c_iter     5 \
-topk       5 \
-acc        0
