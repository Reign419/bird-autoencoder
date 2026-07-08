# Experiment Plan: Dense vs Spatial Latent Bottlenecks

The current project studies whether reconstruction blur is mainly caused by flattening encoder feature maps into Dense latent vectors.

## Main hypothesis

Dense bottleneck:

```text
image -> encoder -> Flatten -> Dense vector -> decoder
```

Spatial bottleneck:

```text
image -> encoder -> H x W x C latent map -> decoder
```

The spatial bottleneck keeps all information inside one latent layer, but preserves spatial structure. It does not use U-Net skip connections.

## Priority experiments

### 1. Same effective latent size

Compare:

| Model | Latent | Size |
|---|---|---|
| residual_lite | 256 | 256 |
| spatial_lite | 8x8x4 | 256 |
| residual_lite | 512 | 512 |
| spatial_lite | 8x8x8 | 512 |

Metrics:

```text
MSE, L1, SSIM, Edge, PSNR
validation reconstruction
validation difference map
```

### 2. Spatial capacity ablation

Compare:

```text
8x8x2
8x8x4
8x8x8
8x8x16
```

### 3. Spatial resolution ablation

Keep effective size fixed:

```text
8x8x4
4x4x16
2x2x64
```

This tests whether spatial layout itself improves reconstruction.

### 4. Loss ablation

Fix the best spatial model and compare:

```text
MSE
L1
L1+SSIM
L1+SSIM+Edge
```

## Future causal representation direction

A likely architecture after these baselines:

```text
image
 -> spatial latent map
       -> decoder (reconstruction)
       -> pooling/concept head (interpretable concepts)
```

This connects reconstruction quality with future concept bottleneck and causal representation learning experiments.
