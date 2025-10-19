# Scaling Field Transformed HMC

Critical slowing down plagues lattice gauge simulations as one approaches the continuum limit. This project implements a field transformed Hybrid Monte Carlo (HMC) algorithm using neural networks to mitigate critical slowing down in 2D U(1) lattice gauge theory simulations.

## Overview

The field transformation approach uses convolutional neural networks to learn optimal field transformations that accelerate the sampling of topological modes in lattice gauge theory. The method trains on gauge configurations at one coupling (beta) and lattice size, then applies the learned transformation to different physical parameters to study scaling behavior.

## Project Structure

```
Scaling_FT_HMC/
├── gauge_generation/          # Generate gauge configurations using standard HMC
├── model_training/            # Train neural network field transformations
├── scaling/                   # Scaling studies and cross-parameter evaluation (Base model batch size = 64)
├── hmc_tune/                  # HMC parameter tuning utilities
├── analysis/                  # Data analysis and summary statistics
├── utils/                     # Core utilities and model definitions
├── *_evaluation/              # Model-specific evaluation directories
│   ├── attn_evaluation/       # Attention model evaluation
│   ├── base_evaluation/       # Base model (batch size = 32) evaluation  
│   ├── combined_evaluation/   # Combined model (batch size = 32) evaluation
│   ├── combined64_evaluation/ # Combined model (batch size = 64) evaluation
│   ├── resn_evaluation/       # ResNet model evaluation
│   └── tanh_evaluation/       # Tanh model evaluation
└── README.md
```

## Neural Network Models

The project implements several CNN architectures for field transformation:

- **base**: Simple 2-layer CNN with GELU activation
- **tanh**: Base model with split tanh output scaling, larger weight for plaquette term
- **resn**: ResNet-style model with residual connections
- **attn**: Attention-enhanced model with channel attention mechanism
- **combined**: Combined architecture incorporating multiple techniques

All models use circular padding to respect lattice periodicity and are designed with small receptive fields to maintain locality properties.

## Installation

### Requirements

This project requires Python 3.9+ and the packages listed in `requirements.txt`. The main dependencies include:

- PyTorch 2.5.1+ with CUDA support
- Lightning for distributed training
- NumPy, SciPy for scientific computing
- Matplotlib, Seaborn for visualization
- gvar for error analysis
- tqdm for progress tracking

### Setup

1. **Clone the repository:**
```bash
git clone -b main --single-branch https://github.com/Greyyy-HJC/Scaling_FT_HMC.git
cd Scaling_FT_HMC
```

2. **Create and activate a conda environment:**
```bash
conda create -n fthmc python=3.9
conda activate fthmc
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
Note that the GPU related dependencies are not installed by default, they are not necessary for analysis. You can uncomment the lines in `requirements.txt` to install them.

4. **Update the path in Python scripts:**
Edit the path in each Python script to point to your local repository:
```python
# sys.path.append('/path/to/your/local')  # replace with the local path of your cloned GitHub repo
```

## Workflow

### 1. Gauge Configuration Generation
```bash
cd gauge_generation
python generate.py --lattice_size 32 --beta 3.0 --n_configs 2048
```

### 2. Model Training
```bash
cd model_training  
python train.py --lattice_size 32 --min_beta 3.0 --max_beta 3.0 --beta_gap 0.5 --model_tag base --save_tag base_train_b3.0_L32 --rand_seed 1029
```

### 3. Evaluation and Scaling Studies
```bash
cd scaling
python compare_fthmc.py --lattice_size 32 --beta 6.0 --train_beta 3.0 --model_tag base --save_tag base_train_b3.0_L32 --rand_seed 1029
python compare_hmc.py --lattice_size 32 --beta 6.0 --rand_seed 1029  # Standard HMC baseline
```

### 4. Analysis
```bash
cd analysis
python summary.py  # Generate comprehensive performance analysis
```

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{anonymous_2024,
  title={TBD},
  author={[Author Names Redacted for Anonymous Submission]},
  journal={TBD},
  year={TBD}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

This research code is provided as-is for reproducibility and educational purposes. For questions or collaboration opportunities, please open an issue.
