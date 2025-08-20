# %%
import os
import torch
import numpy as np
import argparse
import time
import datetime
from Scaling_FT_HMC.utils.func import set_seed
from Scaling_FT_HMC.utils.field_trans import FieldTransformation
from lightning.fabric import Fabric

# Record program start time
start_time = time.time()

# Fabric initialization
fabric = Fabric(accelerator="cuda", strategy="ddp", devices="auto")
fabric.launch()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Training parameters for Field Transformation')
parser.add_argument('--lattice_size', type=int, default=32,
                    help='Size of the lattice (default: 32)')
parser.add_argument('--min_beta', type=float, required=True,
                    help='Minimum beta value for training')
parser.add_argument('--max_beta', type=float, required=True,
                    help='Maximum beta value for training (exclusive)')
parser.add_argument('--beta_gap', type=float, required=True,
                    help='Beta gap for training')
parser.add_argument('--continue_beta', type=float, default=None,
                    help='Continue training from the best model at this beta (default: None)')
parser.add_argument('--n_epochs', type=int, default=16,
                    help='Number of training epochs (default: 16)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training (default: 32)')
parser.add_argument('--n_subsets', type=int, default=8,
                    help='Number of subsets for training (default: 8)')
parser.add_argument('--n_workers', type=int, default=0,
                    help='Number of workers for training (default: 0)')
parser.add_argument('--model_tag', type=str, default='simple',
                    help='Model tag for training (default: simple)')
parser.add_argument('--save_tag', type=str, default=None,
                    help='Save tag for training (default: None), model name; train on which L; which random seed; if test')
parser.add_argument('--rand_seed', type=int, default=1331,
                    help='Random seed for training (default: 1331)')
parser.add_argument('--if_identity_init', action='store_true',
                    help='Initialize models to produce identity transformation (default: False)')
parser.add_argument('--if_check_jac', action='store_true',
                    help='Check Jacobian for training (default: False)')
parser.add_argument('--lr', type=float, default=None,
                    help='Learning rate for training (default: None)')
parser.add_argument('--weight_decay', type=float, default=None,
                    help='Weight decay for training (default: None)')
parser.add_argument('--init_std', type=float, default=None,
                    help='Initial standard deviation for training (default: None)')

args = parser.parse_args()
lattice_size = args.lattice_size

hyperparams = {}
if args.lr is not None:
    hyperparams['lr'] = args.lr
if args.weight_decay is not None:
    hyperparams['weight_decay'] = args.weight_decay
if args.init_std is not None:
    hyperparams['init_std'] = args.init_std

# Print all arguments
fabric.print("="*60)
fabric.print(f">>> CUDA device count: {torch.cuda.device_count()}")
fabric.print(f"PyTorch version: {torch.__version__}")
fabric.print(f"CUDA available: {torch.cuda.is_available()}")
fabric.print(">>> Arguments:")
fabric.print(f"Lattice size: {lattice_size}x{lattice_size}")
fabric.print(f"Minimum beta: {args.min_beta}")
fabric.print(f"Maximum beta: {args.max_beta}")
fabric.print(f"Beta gap: {args.beta_gap}")
fabric.print(f"Continue training from beta: {args.continue_beta}")
fabric.print(f"Number of epochs: {args.n_epochs}")
fabric.print(f"Batch size: {args.batch_size}")
fabric.print(f"Number of subsets: {args.n_subsets}")
fabric.print(f"Number of workers: {args.n_workers}")
fabric.print(f"Model tag: {args.model_tag}")
fabric.print(f"Save tag: {args.save_tag}")
fabric.print(f"Random seed: {args.rand_seed}")
fabric.print(f"Identity initialization: {args.if_identity_init}")
fabric.print(f"Check Jacobian: {args.if_check_jac}")
fabric.print(f"Hyperparameters: {hyperparams}")
fabric.print("="*60)

if fabric.global_rank == 0: # only rank 0 can create directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This program requires GPU to run.")
device = 'cuda'
    
# Set random seed
set_seed(args.rand_seed)

# Set default type
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')

# %%
# initialize the field transformation
nn_ft = FieldTransformation(lattice_size, device=device, n_subsets=args.n_subsets, if_check_jac=args.if_check_jac, num_workers=args.n_workers, identity_init=args.if_identity_init, model_tag=args.model_tag, save_tag=args.save_tag, fabric=fabric, backend='eager', input_hyperparams=hyperparams)

if args.continue_beta is not None:
    continue_beta = args.continue_beta
    nn_ft._load_best_model(train_beta=continue_beta)
    fabric.print(f">>> Loaded the best model at beta = {continue_beta} to continue training")
else:
    fabric.print(">>> Training from scratch")

for train_beta in np.arange(args.min_beta, args.max_beta + args.beta_gap, args.beta_gap):
    beta_start_time = time.time()
    
    # load the data
    data = np.load(f'../gauges/theta_L{lattice_size}_beta{train_beta}.npy')
    
    results = []
    for data_fraction in [1.0, 0.9, 0.8, 0.7]:
        fabric.print(f"\n>>> Training with data fraction = {data_fraction}")
        
        cut_len = int(data_fraction * len(data))
        data = data[:cut_len] # cut off the data
        n_epochs_cut = round(args.n_epochs / data_fraction)
        
        # shuffle the data
        idx = np.random.RandomState(args.rand_seed).permutation(len(data))
        data = data[idx]
        tensor_data = torch.from_numpy(data).float().to(device)
        fabric.print(f"Loaded data shape: {tensor_data.shape}")

        # split the data into training and testing
        train_size = int(0.8 * len(tensor_data))
        train_data = tensor_data[:train_size]
        test_data = tensor_data[train_size:]
        fabric.print(f"Training data shape: {train_data.shape}")
        fabric.print(f"Testing data shape: {test_data.shape}")

        # train the model
        fabric.print("\n>>> Training the model at beta = ", train_beta)
        nn_ft.train(train_data, test_data, train_beta, n_epochs=n_epochs_cut, batch_size=args.batch_size)
        
        # ====== 轻量评估并汇总 ======
        def avg_loss_over_dataset(ft, dataset):
            # eval 模式 + 不打乱，和测试时的做法一致
            ft._set_models_mode(False)
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.n_workers)
            # 可选：如果你担心 DDP 下 dataloader 必须走 fabric，就放开下一行；不想改也能跑
            loader = fabric.setup_dataloaders(loader)

            losses = []
            for batch in loader:
                batch = batch.to(device)
                batch.requires_grad_(True)        # 你的 loss 里需要 autograd
                losses.append(ft.loss_fn(batch).item())
            return float(np.mean(losses)) if losses else float("nan")

        train_avg = avg_loss_over_dataset(nn_ft, train_data)
        test_avg  = avg_loss_over_dataset(nn_ft, test_data)

        results.append({
            "beta": float(train_beta),
            "fraction": float(data_fraction),
            "epochs": int(n_epochs_cut),
            "n_train": int(len(train_data)),
            "n_test": int(len(test_data)),
            "train_loss": train_avg,
            "test_loss": test_avg,
        })
    
    # Calculate and print timing information
    beta_time = time.time() - beta_start_time
    total_time = time.time() - start_time
    
    # Format times as HH:MM:SS
    beta_time_formatted = str(datetime.timedelta(seconds=int(beta_time)))
    total_time_formatted = str(datetime.timedelta(seconds=int(total_time)))
    
    fabric.print(f"\n>>> Completed beta = {train_beta}")
    fabric.print(f">>> Time for this beta: {beta_time_formatted}")
    fabric.print(f">>> Total elapsed time: {total_time_formatted}")
    fabric.print("="*50)
    
    # ===== 打印与落盘当前 beta 的汇总 =====
    if results:
        # 排序一下，按 fraction 从大到小看起来更直观
        results.sort(key=lambda x: x["fraction"], reverse=True)

        fabric.print("\n[Fraction Summary]")
        header = f"{'beta':>5}  {'frac':>4}  {'epochs':>6}  {'n_train':>7}  {'n_test':>6}  {'train_loss':>12}  {'test_loss':>11}"
        fabric.print(header)
        for r in results:
            fabric.print(f"{r['beta']:5.2f}  {r['fraction']:>4.1f}  {r['epochs']:6d}  "
                        f"{r['n_train']:7d}  {r['n_test']:6d}  "
                        f"{r['train_loss']:12.6f}  {r['test_loss']:11.6f}")


