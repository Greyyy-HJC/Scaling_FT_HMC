# %%
import numpy as np
import sys
import gvar as gv
sys.path.append('/eagle/fthmc/run')  # replace with the local path of your cloned GitHub repo
from Scaling_FT_HMC.utils.func import auto_from_chi
from Scaling_FT_HMC.utils.plot_settings import *
from Scaling_FT_HMC.utils.resampling import jackknife, jk_ls_avg

n_steps = 10
rand_seed_ls = [1029, 1107, 1331, 1984, 1999, 2008, 2017, 2025]

# %%
#! base b6 L32

fthmc_base_L32_b6_topo = {}
fthmc_base_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_base_L32_b6_topo[rand_seed] = np.loadtxt(f'../scaling/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_base_train_b3.0_L32_{rand_seed}.csv')
    fthmc_base_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_base_L32_b6_topo[rand_seed][i] - fthmc_base_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_base_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_base_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_base_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_base_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_base_L32_b6_auto_arr = np.array([fthmc_base_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_base_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_base_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_base_L32_b6_deltaQ_arr = np.array([fthmc_base_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_base_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_base_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_base_L32_b6_auto_avg[16])
gamma_ratio_base_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_base_L32_b6 = fthmc_base_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for base L32 b6: {gamma_ratio_base_L32_b6}")
print(f"deltaQ ratio for base L32 b6: {deltaQ_ratio_base_L32_b6}")

# %%
#! base b6 L64

fthmc_base_L64_b6_topo = {}
fthmc_base_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_base_L64_b6_topo[rand_seed] = np.loadtxt(f'../scaling/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_base_train_b3.0_L32_{rand_seed}.csv')
    fthmc_base_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_base_L64_b6_topo[rand_seed][i] - fthmc_base_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_base_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_base_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_base_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_base_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_base_L64_b6_auto_arr = np.array([fthmc_base_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_base_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_base_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_base_L64_b6_deltaQ_arr = np.array([fthmc_base_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_base_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_base_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_base_L64_b6_auto_avg[16])
gamma_ratio_base_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_base_L64_b6 = fthmc_base_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for base L64 b6: {gamma_ratio_base_L64_b6}")
print(f"deltaQ ratio for base L64 b6: {deltaQ_ratio_base_L64_b6}")


# %%
#! arctan b6 L32

fthmc_arctan_L32_b6_topo = {}
fthmc_arctan_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_arctan_L32_b6_topo[rand_seed] = np.loadtxt(f'../arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_arctan_train_b3.0_L32_{rand_seed}.csv')
    fthmc_arctan_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_arctan_L32_b6_topo[rand_seed][i] - fthmc_arctan_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_arctan_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_arctan_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_arctan_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_arctan_L32_b6_auto_arr = np.array([fthmc_arctan_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_arctan_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_arctan_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_arctan_L32_b6_deltaQ_arr = np.array([fthmc_arctan_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_arctan_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_arctan_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_arctan_L32_b6_auto_avg[16])
gamma_ratio_arctan_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_arctan_L32_b6 = fthmc_arctan_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for arctan L32 b6: {gamma_ratio_arctan_L32_b6}")
print(f"deltaQ ratio for arctan L32 b6: {deltaQ_ratio_arctan_L32_b6}")


# %%
#! arctan b6 L64

fthmc_arctan_L64_b6_topo = {}
fthmc_arctan_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_arctan_L64_b6_topo[rand_seed] = np.loadtxt(f'../arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_arctan_train_b3.0_L32_{rand_seed}.csv')
    fthmc_arctan_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_arctan_L64_b6_topo[rand_seed][i] - fthmc_arctan_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_arctan_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_arctan_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_arctan_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_arctan_L64_b6_auto_arr = np.array([fthmc_arctan_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_arctan_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_arctan_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_arctan_L64_b6_deltaQ_arr = np.array([fthmc_arctan_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_arctan_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_arctan_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_arctan_L64_b6_auto_avg[16])
gamma_ratio_arctan_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_arctan_L64_b6 = fthmc_arctan_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for arctan L64 b6: {gamma_ratio_arctan_L64_b6}")
print(f"deltaQ ratio for arctan L64 b6: {deltaQ_ratio_arctan_L64_b6}")


# %%
#! allp b6 L32

fthmc_allp_L32_b6_topo = {}
fthmc_allp_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_allp_L32_b6_topo[rand_seed] = np.loadtxt(f'../allp_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_allp_train_b3.0_L32_{rand_seed}.csv')
    fthmc_allp_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_allp_L32_b6_topo[rand_seed][i] - fthmc_allp_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_allp_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_allp_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_allp_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_allp_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_allp_L32_b6_auto_arr = np.array([fthmc_allp_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_allp_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_allp_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_allp_L32_b6_deltaQ_arr = np.array([fthmc_allp_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_allp_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_allp_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_allp_L32_b6_auto_avg[16])
gamma_ratio_allp_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_allp_L32_b6 = fthmc_allp_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for allp L32 b6: {gamma_ratio_allp_L32_b6}")
print(f"deltaQ ratio for allp L32 b6: {deltaQ_ratio_allp_L32_b6}")

# %%
#! allp b6 L64

fthmc_allp_L64_b6_topo = {}
fthmc_allp_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_allp_L64_b6_topo[rand_seed] = np.loadtxt(f'../allp_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_allp_train_b3.0_L32_{rand_seed}.csv')
    fthmc_allp_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_allp_L64_b6_topo[rand_seed][i] - fthmc_allp_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_allp_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_allp_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_allp_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_allp_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_allp_L64_b6_auto_arr = np.array([fthmc_allp_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_allp_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_allp_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_allp_L64_b6_deltaQ_arr = np.array([fthmc_allp_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_allp_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_allp_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_allp_L64_b6_auto_avg[16])
gamma_ratio_allp_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_allp_L64_b6 = fthmc_allp_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for allp L64 b6: {gamma_ratio_allp_L64_b6}")
print(f"deltaQ ratio for allp L64 b6: {deltaQ_ratio_allp_L64_b6}")


# %%
#! allr b6 L32

fthmc_allr_L32_b6_topo = {}
fthmc_allr_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_allr_L32_b6_topo[rand_seed] = np.loadtxt(f'../allr_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_allr_train_b3.0_L32_{rand_seed}.csv')
    fthmc_allr_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_allr_L32_b6_topo[rand_seed][i] - fthmc_allr_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_allr_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_allr_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_allr_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_allr_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_allr_L32_b6_auto_arr = np.array([fthmc_allr_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_allr_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_allr_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_allr_L32_b6_deltaQ_arr = np.array([fthmc_allr_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_allr_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_allr_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_allr_L32_b6_auto_avg[16])
gamma_ratio_allr_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_allr_L32_b6 = fthmc_allr_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for allr L32 b6: {gamma_ratio_allr_L32_b6}")
print(f"deltaQ ratio for allr L32 b6: {deltaQ_ratio_allr_L32_b6}")

# %%
#! allr b6 L64

fthmc_allr_L64_b6_topo = {}
fthmc_allr_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_allr_L64_b6_topo[rand_seed] = np.loadtxt(f'../allr_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_allr_train_b3.0_L32_{rand_seed}.csv')
    fthmc_allr_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_allr_L64_b6_topo[rand_seed][i] - fthmc_allr_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_allr_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_allr_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_allr_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_allr_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_allr_L64_b6_auto_arr = np.array([fthmc_allr_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_allr_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_allr_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_allr_L64_b6_deltaQ_arr = np.array([fthmc_allr_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_allr_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_allr_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_allr_L64_b6_auto_avg[16])
gamma_ratio_allr_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_allr_L64_b6 = fthmc_allr_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for allr L64 b6: {gamma_ratio_allr_L64_b6}")
print(f"deltaQ ratio for allr L64 b6: {deltaQ_ratio_allr_L64_b6}")




# %%
#! 2plaq b6 L32

fthmc_2plaq_L32_b6_topo = {}
fthmc_2plaq_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_2plaq_L32_b6_topo[rand_seed] = np.loadtxt(f'../2plaq_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_2plaq_train_b3.0_L32_{rand_seed}.csv')
    fthmc_2plaq_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_2plaq_L32_b6_topo[rand_seed][i] - fthmc_2plaq_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_2plaq_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_2plaq_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_2plaq_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_2plaq_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_2plaq_L32_b6_auto_arr = np.array([fthmc_2plaq_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_2plaq_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_2plaq_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_2plaq_L32_b6_deltaQ_arr = np.array([fthmc_2plaq_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_2plaq_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_2plaq_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_2plaq_L32_b6_auto_avg[16])
gamma_ratio_2plaq_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_2plaq_L32_b6 = fthmc_2plaq_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for 2plaq L32 b6: {gamma_ratio_2plaq_L32_b6}")
print(f"deltaQ ratio for 2plaq L32 b6: {deltaQ_ratio_2plaq_L32_b6}")

# %%
#! 2plaq b6 L64

fthmc_2plaq_L64_b6_topo = {}
fthmc_2plaq_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_2plaq_L64_b6_topo[rand_seed] = np.loadtxt(f'../2plaq_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_2plaq_train_b3.0_L32_{rand_seed}.csv')
    fthmc_2plaq_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_2plaq_L64_b6_topo[rand_seed][i] - fthmc_2plaq_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_2plaq_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_2plaq_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_2plaq_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_2plaq_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_2plaq_L64_b6_auto_arr = np.array([fthmc_2plaq_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_2plaq_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_2plaq_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_2plaq_L64_b6_deltaQ_arr = np.array([fthmc_2plaq_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_2plaq_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_2plaq_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_2plaq_L64_b6_auto_avg[16])
gamma_ratio_2plaq_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_2plaq_L64_b6 = fthmc_2plaq_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for 2plaq L64 b6: {gamma_ratio_2plaq_L64_b6}")
print(f"deltaQ ratio for 2plaq L64 b6: {deltaQ_ratio_2plaq_L64_b6}")




# %%
#! 2plaq_weight b6 L32

fthmc_2plaq_weight_L32_b6_topo = {}
fthmc_2plaq_weight_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_2plaq_weight_L32_b6_topo[rand_seed] = np.loadtxt(f'../2plaq_weight_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_2plaq_weight_train_b3.0_L32_{rand_seed}.csv')
    fthmc_2plaq_weight_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_2plaq_weight_L32_b6_topo[rand_seed][i] - fthmc_2plaq_weight_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_2plaq_weight_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_2plaq_weight_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_2plaq_weight_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_2plaq_weight_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_2plaq_weight_L32_b6_auto_arr = np.array([fthmc_2plaq_weight_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_2plaq_weight_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_2plaq_weight_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_2plaq_weight_L32_b6_deltaQ_arr = np.array([fthmc_2plaq_weight_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_2plaq_weight_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_2plaq_weight_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_2plaq_weight_L32_b6_auto_avg[16])
gamma_ratio_2plaq_weight_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_2plaq_weight_L32_b6 = fthmc_2plaq_weight_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for 2plaq_weight L32 b6: {gamma_ratio_2plaq_weight_L32_b6}")
print(f"deltaQ ratio for 2plaq_weight L32 b6: {deltaQ_ratio_2plaq_weight_L32_b6}")

# %%
#! 2plaq_weight b6 L64

fthmc_2plaq_weight_L64_b6_topo = {}
fthmc_2plaq_weight_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_2plaq_weight_L64_b6_topo[rand_seed] = np.loadtxt(f'../2plaq_weight_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_2plaq_weight_train_b3.0_L32_{rand_seed}.csv')
    fthmc_2plaq_weight_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_2plaq_weight_L64_b6_topo[rand_seed][i] - fthmc_2plaq_weight_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_2plaq_weight_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_2plaq_weight_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_2plaq_weight_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_2plaq_weight_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_2plaq_weight_L64_b6_auto_arr = np.array([fthmc_2plaq_weight_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_2plaq_weight_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_2plaq_weight_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_2plaq_weight_L64_b6_deltaQ_arr = np.array([fthmc_2plaq_weight_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_2plaq_weight_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_2plaq_weight_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_2plaq_weight_L64_b6_auto_avg[16])
gamma_ratio_2plaq_weight_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_2plaq_weight_L64_b6 = fthmc_2plaq_weight_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for 2plaq_weight L64 b6: {gamma_ratio_2plaq_weight_L64_b6}")
print(f"deltaQ ratio for 2plaq_weight L64 b6: {deltaQ_ratio_2plaq_weight_L64_b6}")




# %%
#! tanh b6 L32

fthmc_tanh_L32_b6_topo = {}
fthmc_tanh_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_tanh_L32_b6_topo[rand_seed] = np.loadtxt(f'../tanh_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_tanh_train_b3.0_L32_{rand_seed}.csv')
    fthmc_tanh_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_tanh_L32_b6_topo[rand_seed][i] - fthmc_tanh_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_tanh_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_tanh_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_tanh_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_tanh_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_tanh_L32_b6_auto_arr = np.array([fthmc_tanh_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_tanh_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_tanh_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_tanh_L32_b6_deltaQ_arr = np.array([fthmc_tanh_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_tanh_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_tanh_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_tanh_L32_b6_auto_avg[16])
gamma_ratio_tanh_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_tanh_L32_b6 = fthmc_tanh_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for tanh L32 b6: {gamma_ratio_tanh_L32_b6}")
print(f"deltaQ ratio for tanh L32 b6: {deltaQ_ratio_tanh_L32_b6}")

# %%
#! tanh b6 L64

fthmc_tanh_L64_b6_topo = {}
fthmc_tanh_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_tanh_L64_b6_topo[rand_seed] = np.loadtxt(f'../tanh_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_tanh_train_b3.0_L32_{rand_seed}.csv')
    fthmc_tanh_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_tanh_L64_b6_topo[rand_seed][i] - fthmc_tanh_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_tanh_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_tanh_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_tanh_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_tanh_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_tanh_L64_b6_auto_arr = np.array([fthmc_tanh_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_tanh_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_tanh_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_tanh_L64_b6_deltaQ_arr = np.array([fthmc_tanh_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_tanh_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_tanh_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_tanh_L64_b6_auto_avg[16])
gamma_ratio_tanh_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_tanh_L64_b6 = fthmc_tanh_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for tanh L64 b6: {gamma_ratio_tanh_L64_b6}")
print(f"deltaQ ratio for tanh L64 b6: {deltaQ_ratio_tanh_L64_b6}")



# %%
#! summary

gamma_L32_b6_ratio_ls = [gamma_ratio_tanh_L32_b6, gamma_ratio_arctan_L32_b6, gamma_ratio_allp_L32_b6, gamma_ratio_allr_L32_b6, gamma_ratio_2plaq_L32_b6, gamma_ratio_2plaq_weight_L32_b6]

deltaQ_L32_b6_ratio_ls = [deltaQ_ratio_tanh_L32_b6, deltaQ_ratio_arctan_L32_b6, deltaQ_ratio_allp_L32_b6, deltaQ_ratio_allr_L32_b6, deltaQ_ratio_2plaq_L32_b6, deltaQ_ratio_2plaq_weight_L32_b6]

gamma_L64_b6_ratio_ls = [gamma_ratio_tanh_L64_b6, gamma_ratio_arctan_L64_b6, gamma_ratio_allp_L64_b6, gamma_ratio_allr_L64_b6, gamma_ratio_2plaq_L64_b6, gamma_ratio_2plaq_weight_L64_b6]

deltaQ_L64_b6_ratio_ls = [deltaQ_ratio_tanh_L64_b6, deltaQ_ratio_arctan_L64_b6, deltaQ_ratio_allp_L64_b6, deltaQ_ratio_allr_L64_b6, deltaQ_ratio_2plaq_L64_b6, deltaQ_ratio_2plaq_weight_L64_b6]


fig, ax = default_plot()

ax.errorbar(np.arange(len(gamma_L32_b6_ratio_ls))-0.05, [gv.mean(gamma_ratio) for gamma_ratio in gamma_L32_b6_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_L32_b6_ratio_ls], label="$\\beta=6$, $V=32^2$", marker="x", **errorb)

ax.errorbar(np.arange(len(gamma_L64_b6_ratio_ls))+0.05, [gv.mean(gamma_ratio) for gamma_ratio in gamma_L64_b6_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_L64_b6_ratio_ls], label="$\\beta=6$, $V=64^2$", marker="x", **errorb)

ax.set_ylabel('$\\gamma (16)_{\\mathrm{Base}} ~/~ \\gamma (16)$', **fs_p)
ax.set_ylim(0.5, 1.6)
ax.set_xticks(np.arange(len(gamma_L64_b6_ratio_ls)))
ax.set_xticklabels(['Tanh', 'arctan', 'allp', 'allr', 'plaq2', 'plaq2_wt'], **fs_p)
ax.legend(ncol=2, loc='upper right', **fs_small_p)
plt.tight_layout()
plt.savefig('plots/test_summary_train_b3_L32_gamma.pdf', transparent=True)
plt.show()


fig, ax = default_plot()

ax.errorbar(np.arange(len(deltaQ_L32_b6_ratio_ls))-0.05, [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_L32_b6_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_L32_b6_ratio_ls], label="$\\beta=6$, $V=32^2$", **errorb_circle)

ax.errorbar(np.arange(len(deltaQ_L64_b6_ratio_ls))+0.05, [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_L64_b6_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_L64_b6_ratio_ls], label="$\\beta=6$, $V=64^2$", **errorb_circle)

ax.set_ylabel('$\\Delta Q ~/~ \\Delta Q_{\\mathrm{Base}}$', **fs_p)
ax.set_ylim(0.5, 1.6)
ax.set_xticks(np.arange(len(deltaQ_L64_b6_ratio_ls)))
ax.set_xticklabels(['Tanh', 'arctan', 'allp', 'allr', 'plaq2', 'plaq2_wt'], **fs_p)
ax.legend(ncol=2, loc='upper right', **fs_small_p)
plt.tight_layout()
plt.savefig('plots/test_summary_train_b3_L32_deltaQ.pdf', transparent=True)
plt.show()

# %%
