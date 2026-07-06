# Bird Image Autoencoder

This repository contains convolutional, residual, and ResNet-based autoencoder baselines for image reconstruction on the CUB-200-2011 bird dataset.

## Goal

The goal is to learn a compact single-vector latent representation and reconstruct the input image.

Pipeline:

```text
image -> encoder -> single latent vector -> decoder -> reconstructed image
