# Scaling Field Transformed HMC

Critical slowing down plagues lattice gauge simulations as one approaches the continuum limit. 

This project implements a field transformed HMC algorithm to reduce the critical slowing down. 

## Folder Structure

```
Scaling_FT_HMC/
    gauge_generation/
    model_training/
    scaling/
    utils/
```

## Logs

### Scaling

- Take base model, train on beta=3.0, L=32, apply on beta in [3, 4, 5, 6] and L in [32, 64] (maybe 128 ?)

### Comparison of models

- Train on beta=3.0, L=32
- Apply on beta=3.0, L=32; beta=6.0, L=32; and beta=6.0, L=64

- Train on beta=4.0, L=64
- Apply on beta=4.0, L=64; beta=6.0, L=32; and beta=6.0, L=64