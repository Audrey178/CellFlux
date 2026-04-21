#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
python train.py \
    --dataset=ctpet \
    --config=ctpet \
    --batch_size=1 \
    --accum_iter=2 \
    --eval_frequency=10 \
    --epochs=10 \
    --class_drop_prob=1.0 \
    --cfg_scale=0.0 \
    --ode_method heun2 \
    --ode_options '{"nfe": 50}' \
    --use_ema \
    --edm_schedule \
    --skewed_timesteps \
    --compute_recon_metrics \
    --use_initial=2 \
    --noise_level=0.5 \
    --output_dir=outputs/example \
    --save_visualizations \
    --num_visual_samples=16 \
    --test_split=0.1 \
