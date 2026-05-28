This repository contains Python scripts for generating VAE-based latent representations and using them for disease stage classification.

## Workflow

1. Run a VAE script to generate latent vectors:

vae180_sum.py
vae-9.py
vae_merge.py

These scripts save:

vae_latents.npy
vae_labels.npy
vae_subject_ids.npy
Run classifier/evaluation scripts on the saved latent vectors:
mlp-2.py
mlp_tuning.py
logistic_reg.py

The classifiers evaluate the latent representations using MLP, KNN, subject-based splits, K-fold evaluation, accuracy, F1 score, confusion matrices, and visualization.

## Technologies:

Python
PyTorch
NumPy
scikit-learn
Matplotlib
SciPy
