#!/bin/bash
python3 -m WrappingNet --gpus 0 --epochs 100 --epochs_sphere 200 --latent_dim 2048 --lr 1e-5 --data_name "manifold40" --model_name "LC"
