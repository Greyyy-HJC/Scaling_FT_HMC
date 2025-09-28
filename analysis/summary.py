# %%
import numpy as np
import sys
import gvar as gv
# sys.path.append('/path/to/your/local')  # replace with the local path of your cloned GitHub repo
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
#! base b7 L128

fthmc_base_L128_b7_topo = {}
fthmc_base_L128_b7_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_base_L128_b7_topo[rand_seed] = np.loadtxt(f'../scaling/dumps/topo_fthmc_L128_beta7.0_nsteps{n_steps}_base_train_b3.0_L32_{rand_seed}.csv')
    fthmc_base_L128_b7_deltaQ[rand_seed] = np.mean([ abs(fthmc_base_L128_b7_topo[rand_seed][i] - fthmc_base_L128_b7_topo[rand_seed][i-1]) for i in range(1, len(fthmc_base_L128_b7_topo[rand_seed]))])

beta = 7.0
max_lag = 20
volume = 128**2

fthmc_base_L128_b7_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_base_L128_b7_auto[rand_seed] = auto_from_chi(fthmc_base_L128_b7_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_base_L128_b7_auto_arr = np.array([fthmc_base_L128_b7_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L128_b7_auto_arr, fthmc_base_L128_b7_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L128_b7_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_base_L128_b7_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_base_L128_b7_deltaQ_arr = np.array([fthmc_base_L128_b7_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L128_b7_deltaQ_arr, fthmc_base_L128_b7_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L128_b7_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_base_L128_b7_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L128_b7_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_base_L128_b7_auto_avg[16])
gamma_ratio_base_L128_b7 = gamma_base / gamma_fthmc
deltaQ_ratio_base_L128_b7 = fthmc_base_L128_b7_deltaQ_avg / base_L128_b7_deltaQ_avg

print(f"gamma ratio for base L128 b7: {gamma_ratio_base_L128_b7}")
print(f"deltaQ ratio for base L128 b7: {deltaQ_ratio_base_L128_b7}")


# %%
#! base32 b6 L32

fthmc_base32_L32_b6_topo = {}
fthmc_base32_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_base32_L32_b6_topo[rand_seed] = np.loadtxt(f'../base_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_base_batch32_train_b3.0_L32_{rand_seed}.csv')
    fthmc_base32_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_base32_L32_b6_topo[rand_seed][i] - fthmc_base32_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_base32_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_base32_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_base32_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_base32_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_base32_L32_b6_auto_arr = np.array([fthmc_base32_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_base32_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_base32_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_base32_L32_b6_deltaQ_arr = np.array([fthmc_base32_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_base32_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_base32_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_base32_L32_b6_auto_avg[16])
gamma_ratio_base32_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_base32_L32_b6 = fthmc_base32_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for base32 L32 b6: {gamma_ratio_base32_L32_b6}")
print(f"deltaQ ratio for base32 L32 b6: {deltaQ_ratio_base32_L32_b6}")


# %%
#! base32 b6 L64
fthmc_base32_L64_b6_topo = {}
fthmc_base32_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_base32_L64_b6_topo[rand_seed] = np.loadtxt(f'../base_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_base_batch32_train_b3.0_L32_{rand_seed}.csv')
    fthmc_base32_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_base32_L64_b6_topo[rand_seed][i] - fthmc_base32_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_base32_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_base32_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_base32_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_base32_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_base32_L64_b6_auto_arr = np.array([fthmc_base32_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_base32_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_base32_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_base32_L64_b6_deltaQ_arr = np.array([fthmc_base32_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_base32_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_base32_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_base32_L64_b6_auto_avg[16])
gamma_ratio_base32_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_base32_L64_b6 = fthmc_base32_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for base32 L64 b6: {gamma_ratio_base32_L64_b6}")
print(f"deltaQ ratio for base32 L64 b6: {deltaQ_ratio_base32_L64_b6}")


# %%
#! base32 b7 L128
fthmc_base32_L128_b7_topo = {}
fthmc_base32_L128_b7_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_base32_L128_b7_topo[rand_seed] = np.loadtxt(f'../base_evaluation/dumps/topo_fthmc_L128_beta7.0_nsteps{n_steps}_base_batch32_train_b3.0_L32_{rand_seed}.csv')
    fthmc_base32_L128_b7_deltaQ[rand_seed] = np.mean([ abs(fthmc_base32_L128_b7_topo[rand_seed][i] - fthmc_base32_L128_b7_topo[rand_seed][i-1]) for i in range(1, len(fthmc_base32_L128_b7_topo[rand_seed]))])

beta = 7.0
max_lag = 20
volume = 128**2

fthmc_base32_L128_b7_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_base32_L128_b7_auto[rand_seed] = auto_from_chi(fthmc_base32_L128_b7_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_base32_L128_b7_auto_arr = np.array([fthmc_base32_L128_b7_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L128_b7_auto_arr, fthmc_base32_L128_b7_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L128_b7_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_base32_L128_b7_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_base32_L128_b7_deltaQ_arr = np.array([fthmc_base32_L128_b7_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L128_b7_deltaQ_arr, fthmc_base32_L128_b7_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L128_b7_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_base32_L128_b7_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L128_b7_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_base32_L128_b7_auto_avg[16])
gamma_ratio_base32_L128_b7 = gamma_base / gamma_fthmc
deltaQ_ratio_base32_L128_b7 = fthmc_base32_L128_b7_deltaQ_avg / base_L128_b7_deltaQ_avg

print(f"gamma ratio for base32 L128 b7: {gamma_ratio_base32_L128_b7}")
print(f"deltaQ ratio for base32 L128 b7: {deltaQ_ratio_base32_L128_b7}")


# %%
#! attn b6 L32

fthmc_attn_L32_b6_topo = {}
fthmc_attn_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_attn_L32_b6_topo[rand_seed] = np.loadtxt(f'../attn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_attn_train_b3.0_L32_{rand_seed}.csv')
    fthmc_attn_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_attn_L32_b6_topo[rand_seed][i] - fthmc_attn_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_attn_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_attn_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_attn_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_attn_L32_b6_auto_arr = np.array([fthmc_attn_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_attn_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_attn_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_attn_L32_b6_deltaQ_arr = np.array([fthmc_attn_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_attn_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_attn_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_attn_L32_b6_auto_avg[16])
gamma_ratio_attn_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_attn_L32_b6 = fthmc_attn_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for attn L32 b6: {gamma_ratio_attn_L32_b6}")
print(f"deltaQ ratio for attn L32 b6: {deltaQ_ratio_attn_L32_b6}")


# %%
#! attn b6 L64

fthmc_attn_L64_b6_topo = {}
fthmc_attn_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_attn_L64_b6_topo[rand_seed] = np.loadtxt(f'../attn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_attn_train_b3.0_L32_{rand_seed}.csv')
    fthmc_attn_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_attn_L64_b6_topo[rand_seed][i] - fthmc_attn_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_attn_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_attn_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_attn_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_attn_L64_b6_auto_arr = np.array([fthmc_attn_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_attn_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_attn_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_attn_L64_b6_deltaQ_arr = np.array([fthmc_attn_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_attn_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_attn_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_attn_L64_b6_auto_avg[16])
gamma_ratio_attn_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_attn_L64_b6 = fthmc_attn_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for attn L64 b6: {gamma_ratio_attn_L64_b6}")
print(f"deltaQ ratio for attn L64 b6: {deltaQ_ratio_attn_L64_b6}")


# %%
#! attn b7 L128

fthmc_attn_L128_b7_topo = {}
fthmc_attn_L128_b7_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_attn_L128_b7_topo[rand_seed] = np.loadtxt(f'../attn_evaluation/dumps/topo_fthmc_L128_beta7.0_nsteps{n_steps}_attn_train_b3.0_L32_{rand_seed}.csv')
    fthmc_attn_L128_b7_deltaQ[rand_seed] = np.mean([ abs(fthmc_attn_L128_b7_topo[rand_seed][i] - fthmc_attn_L128_b7_topo[rand_seed][i-1]) for i in range(1, len(fthmc_attn_L128_b7_topo[rand_seed]))])

beta = 7.0
max_lag = 20
volume = 128**2

fthmc_attn_L128_b7_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_attn_L128_b7_auto[rand_seed] = auto_from_chi(fthmc_attn_L128_b7_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_attn_L128_b7_auto_arr = np.array([fthmc_attn_L128_b7_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L128_b7_auto_arr, fthmc_attn_L128_b7_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L128_b7_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_attn_L128_b7_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_attn_L128_b7_deltaQ_arr = np.array([fthmc_attn_L128_b7_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L128_b7_deltaQ_arr, fthmc_attn_L128_b7_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L128_b7_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_attn_L128_b7_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L128_b7_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_attn_L128_b7_auto_avg[16])
gamma_ratio_attn_L128_b7 = gamma_base / gamma_fthmc
deltaQ_ratio_attn_L128_b7 = fthmc_attn_L128_b7_deltaQ_avg / base_L128_b7_deltaQ_avg

print(f"gamma ratio for attn L128 b7: {gamma_ratio_attn_L128_b7}")
print(f"deltaQ ratio for attn L128 b7: {deltaQ_ratio_attn_L128_b7}")



# %%
#! resn b6 L32

fthmc_resn_L32_b6_topo = {}
fthmc_resn_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_resn_L32_b6_topo[rand_seed] = np.loadtxt(f'../resn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_resn_train_b3.0_L32_{rand_seed}.csv')
    fthmc_resn_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_resn_L32_b6_topo[rand_seed][i] - fthmc_resn_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_resn_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_resn_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_resn_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_resn_L32_b6_auto_arr = np.array([fthmc_resn_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_resn_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_resn_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_resn_L32_b6_deltaQ_arr = np.array([fthmc_resn_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_resn_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_resn_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_resn_L32_b6_auto_avg[16])
gamma_ratio_resn_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_resn_L32_b6 = fthmc_resn_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for resn L32 b6: {gamma_ratio_resn_L32_b6}")
print(f"deltaQ ratio for resn L32 b6: {deltaQ_ratio_resn_L32_b6}")

# %%
#! resn b6 L64

fthmc_resn_L64_b6_topo = {}
fthmc_resn_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_resn_L64_b6_topo[rand_seed] = np.loadtxt(f'../resn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_resn_train_b3.0_L32_{rand_seed}.csv')
    fthmc_resn_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_resn_L64_b6_topo[rand_seed][i] - fthmc_resn_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_resn_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_resn_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_resn_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_resn_L64_b6_auto_arr = np.array([fthmc_resn_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_resn_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_resn_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_resn_L64_b6_deltaQ_arr = np.array([fthmc_resn_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_resn_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_resn_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_resn_L64_b6_auto_avg[16])
gamma_ratio_resn_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_resn_L64_b6 = fthmc_resn_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for resn L64 b6: {gamma_ratio_resn_L64_b6}")
print(f"deltaQ ratio for resn L64 b6: {deltaQ_ratio_resn_L64_b6}")

# %%
#! resn b7 L128

fthmc_resn_L128_b7_topo = {}
fthmc_resn_L128_b7_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_resn_L128_b7_topo[rand_seed] = np.loadtxt(f'../resn_evaluation/dumps/topo_fthmc_L128_beta7.0_nsteps{n_steps}_resn_train_b3.0_L32_{rand_seed}.csv')
    fthmc_resn_L128_b7_deltaQ[rand_seed] = np.mean([ abs(fthmc_resn_L128_b7_topo[rand_seed][i] - fthmc_resn_L128_b7_topo[rand_seed][i-1]) for i in range(1, len(fthmc_resn_L128_b7_topo[rand_seed]))])

beta = 7.0
max_lag = 20
volume = 128**2

fthmc_resn_L128_b7_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_resn_L128_b7_auto[rand_seed] = auto_from_chi(fthmc_resn_L128_b7_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_resn_L128_b7_auto_arr = np.array([fthmc_resn_L128_b7_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L128_b7_auto_arr, fthmc_resn_L128_b7_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L128_b7_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_resn_L128_b7_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_resn_L128_b7_deltaQ_arr = np.array([fthmc_resn_L128_b7_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L128_b7_deltaQ_arr, fthmc_resn_L128_b7_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L128_b7_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_resn_L128_b7_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L128_b7_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_resn_L128_b7_auto_avg[16])
gamma_ratio_resn_L128_b7 = gamma_base / gamma_fthmc
deltaQ_ratio_resn_L128_b7 = fthmc_resn_L128_b7_deltaQ_avg / base_L128_b7_deltaQ_avg

print(f"gamma ratio for resn L128 b7: {gamma_ratio_resn_L128_b7}")
print(f"deltaQ ratio for resn L128 b7: {deltaQ_ratio_resn_L128_b7}")


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
#! tanh b7 L128

fthmc_tanh_L128_b7_topo = {}
fthmc_tanh_L128_b7_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_tanh_L128_b7_topo[rand_seed] = np.loadtxt(f'../tanh_evaluation/dumps/topo_fthmc_L128_beta7.0_nsteps{n_steps}_tanh_train_b3.0_L32_{rand_seed}.csv')
    fthmc_tanh_L128_b7_deltaQ[rand_seed] = np.mean([ abs(fthmc_tanh_L128_b7_topo[rand_seed][i] - fthmc_tanh_L128_b7_topo[rand_seed][i-1]) for i in range(1, len(fthmc_tanh_L128_b7_topo[rand_seed]))])

beta = 7.0
max_lag = 20
volume = 128**2

fthmc_tanh_L128_b7_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_tanh_L128_b7_auto[rand_seed] = auto_from_chi(fthmc_tanh_L128_b7_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_tanh_L128_b7_auto_arr = np.array([fthmc_tanh_L128_b7_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L128_b7_auto_arr, fthmc_tanh_L128_b7_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L128_b7_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_tanh_L128_b7_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_tanh_L128_b7_deltaQ_arr = np.array([fthmc_tanh_L128_b7_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L128_b7_deltaQ_arr, fthmc_tanh_L128_b7_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L128_b7_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_tanh_L128_b7_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L128_b7_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_tanh_L128_b7_auto_avg[16])
gamma_ratio_tanh_L128_b7 = gamma_base / gamma_fthmc
deltaQ_ratio_tanh_L128_b7 = fthmc_tanh_L128_b7_deltaQ_avg / base_L128_b7_deltaQ_avg

print(f"gamma ratio for tanh L128 b7: {gamma_ratio_tanh_L128_b7}")
print(f"deltaQ ratio for tanh L128 b7: {deltaQ_ratio_tanh_L128_b7}")



# %%
#! combined b6 L32

fthmc_combined_L32_b6_topo = {}
fthmc_combined_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_combined_L32_b6_topo[rand_seed] = np.loadtxt(f'../combined_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_combined_train_b3.0_L32_{rand_seed}.csv')
    fthmc_combined_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_combined_L32_b6_topo[rand_seed][i] - fthmc_combined_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_combined_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_combined_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_combined_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_combined_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_combined_L32_b6_auto_arr = np.array([fthmc_combined_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_combined_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_combined_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_combined_L32_b6_deltaQ_arr = np.array([fthmc_combined_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_combined_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_combined_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_combined_L32_b6_auto_avg[16])
gamma_ratio_combined_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_combined_L32_b6 = fthmc_combined_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for combined L32 b6: {gamma_ratio_combined_L32_b6}")
print(f"deltaQ ratio for combined L32 b6: {deltaQ_ratio_combined_L32_b6}")


# %%
#! combined b6 L64

fthmc_combined_L64_b6_topo = {}
fthmc_combined_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_combined_L64_b6_topo[rand_seed] = np.loadtxt(f'../combined_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_combined_train_b3.0_L32_{rand_seed}.csv')
    fthmc_combined_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_combined_L64_b6_topo[rand_seed][i] - fthmc_combined_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_combined_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_combined_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_combined_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_combined_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_combined_L64_b6_auto_arr = np.array([fthmc_combined_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_combined_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_combined_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_combined_L64_b6_deltaQ_arr = np.array([fthmc_combined_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_combined_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_combined_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_combined_L64_b6_auto_avg[16])
gamma_ratio_combined_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_combined_L64_b6 = fthmc_combined_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for combined L64 b6: {gamma_ratio_combined_L64_b6}")
print(f"deltaQ ratio for combined L64 b6: {deltaQ_ratio_combined_L64_b6}")


# %%
#! combined b7 L128

fthmc_combined_L128_b7_topo = {}
fthmc_combined_L128_b7_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_combined_L128_b7_topo[rand_seed] = np.loadtxt(f'../combined_evaluation/dumps/topo_fthmc_L128_beta7.0_nsteps{n_steps}_combined_train_b3.0_L32_{rand_seed}.csv')
    fthmc_combined_L128_b7_deltaQ[rand_seed] = np.mean([ abs(fthmc_combined_L128_b7_topo[rand_seed][i] - fthmc_combined_L128_b7_topo[rand_seed][i-1]) for i in range(1, len(fthmc_combined_L128_b7_topo[rand_seed]))])

beta = 7.0
max_lag = 20
volume = 128**2

fthmc_combined_L128_b7_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_combined_L128_b7_auto[rand_seed] = auto_from_chi(fthmc_combined_L128_b7_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_combined_L128_b7_auto_arr = np.array([fthmc_combined_L128_b7_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L128_b7_auto_arr, fthmc_combined_L128_b7_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L128_b7_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_combined_L128_b7_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_combined_L128_b7_deltaQ_arr = np.array([fthmc_combined_L128_b7_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L128_b7_deltaQ_arr, fthmc_combined_L128_b7_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L128_b7_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_combined_L128_b7_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L128_b7_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_combined_L128_b7_auto_avg[16])
gamma_ratio_combined_L128_b7 = gamma_base / gamma_fthmc
deltaQ_ratio_combined_L128_b7 = fthmc_combined_L128_b7_deltaQ_avg / base_L128_b7_deltaQ_avg

print(f"gamma ratio for combined L128 b7: {gamma_ratio_combined_L128_b7}")
print(f"deltaQ ratio for combined L128 b7: {deltaQ_ratio_combined_L128_b7}")


# %%
#! combined64 b6 L32

fthmc_combined64_L32_b6_topo = {}
fthmc_combined64_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_combined64_L32_b6_topo[rand_seed] = np.loadtxt(f'../combined64_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_combined64_train_b3.0_L32_{rand_seed}.csv')
    fthmc_combined64_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_combined64_L32_b6_topo[rand_seed][i] - fthmc_combined64_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_combined64_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_combined64_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_combined64_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_combined64_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_combined64_L32_b6_auto_arr = np.array([fthmc_combined64_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L32_b6_auto_arr, fthmc_combined64_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_combined64_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_combined64_L32_b6_deltaQ_arr = np.array([fthmc_combined64_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L32_b6_deltaQ_arr, fthmc_combined64_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_combined64_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_combined64_L32_b6_auto_avg[16])
gamma_ratio_combined64_L32_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_combined64_L32_b6 = fthmc_combined64_L32_b6_deltaQ_avg / base_L32_b6_deltaQ_avg

print(f"gamma ratio for combined64 L32 b6: {gamma_ratio_combined64_L32_b6}")
print(f"deltaQ ratio for combined64 L32 b6: {deltaQ_ratio_combined64_L32_b6}")

# %%
#! combined64 b6 L64

fthmc_combined64_L64_b6_topo = {}
fthmc_combined64_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_combined64_L64_b6_topo[rand_seed] = np.loadtxt(f'../combined64_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_combined64_train_b3.0_L32_{rand_seed}.csv')
    fthmc_combined64_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_combined64_L64_b6_topo[rand_seed][i] - fthmc_combined64_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_combined64_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_combined64_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_combined64_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_combined64_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_combined64_L64_b6_auto_arr = np.array([fthmc_combined64_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L64_b6_auto_arr, fthmc_combined64_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_combined64_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_combined64_L64_b6_deltaQ_arr = np.array([fthmc_combined64_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L64_b6_deltaQ_arr, fthmc_combined64_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_combined64_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_combined64_L64_b6_auto_avg[16])
gamma_ratio_combined64_L64_b6 = gamma_base / gamma_fthmc
deltaQ_ratio_combined64_L64_b6 = fthmc_combined64_L64_b6_deltaQ_avg / base_L64_b6_deltaQ_avg

print(f"gamma ratio for combined64 L64 b6: {gamma_ratio_combined64_L64_b6}")
print(f"deltaQ ratio for combined64 L64 b6: {deltaQ_ratio_combined64_L64_b6}")


# %%
#! combined64 b7 L128

fthmc_combined64_L128_b7_topo = {}
fthmc_combined64_L128_b7_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_combined64_L128_b7_topo[rand_seed] = np.loadtxt(f'../combined64_evaluation/dumps/topo_fthmc_L128_beta7.0_nsteps{n_steps}_combined64_train_b3.0_L32_{rand_seed}.csv')
    fthmc_combined64_L128_b7_deltaQ[rand_seed] = np.mean([ abs(fthmc_combined64_L128_b7_topo[rand_seed][i] - fthmc_combined64_L128_b7_topo[rand_seed][i-1]) for i in range(1, len(fthmc_combined64_L128_b7_topo[rand_seed]))])

beta = 7.0
max_lag = 20
volume = 128**2

fthmc_combined64_L128_b7_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_combined64_L128_b7_auto[rand_seed] = auto_from_chi(fthmc_combined64_L128_b7_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_combined64_L128_b7_auto_arr = np.array([fthmc_combined64_L128_b7_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([fthmc_base_L128_b7_auto_arr, fthmc_combined64_L128_b7_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
base_L128_b7_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_combined64_L128_b7_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_combined64_L128_b7_deltaQ_arr = np.array([fthmc_combined64_L128_b7_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([fthmc_base_L128_b7_deltaQ_arr, fthmc_combined64_L128_b7_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
base_L128_b7_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_combined64_L128_b7_deltaQ_avg = jk_deltaQ_avg[1]


gamma_base = 1 / (1 - base_L128_b7_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_combined64_L128_b7_auto_avg[16])
gamma_ratio_combined64_L128_b7 = gamma_base / gamma_fthmc
deltaQ_ratio_combined64_L128_b7 = fthmc_combined64_L128_b7_deltaQ_avg / base_L128_b7_deltaQ_avg

print(f"gamma ratio for combined64 L128 b7: {gamma_ratio_combined64_L128_b7}")
print(f"deltaQ ratio for combined64 L128 b7: {deltaQ_ratio_combined64_L128_b7}")



# %%
#! summary

gamma_L32_b6_ratio_ls = [gamma_ratio_base32_L32_b6, gamma_ratio_attn_L32_b6, gamma_ratio_resn_L32_b6, gamma_ratio_tanh_L32_b6, gamma_ratio_combined64_L32_b6, gamma_ratio_combined_L32_b6]

deltaQ_L32_b6_ratio_ls = [deltaQ_ratio_base32_L32_b6, deltaQ_ratio_attn_L32_b6, deltaQ_ratio_resn_L32_b6, deltaQ_ratio_tanh_L32_b6, deltaQ_ratio_combined64_L32_b6, deltaQ_ratio_combined_L32_b6]

gamma_L64_b6_ratio_ls = [gamma_ratio_base32_L64_b6, gamma_ratio_attn_L64_b6, gamma_ratio_resn_L64_b6, gamma_ratio_tanh_L64_b6, gamma_ratio_combined64_L64_b6, gamma_ratio_combined_L64_b6]

deltaQ_L64_b6_ratio_ls = [deltaQ_ratio_base32_L64_b6, deltaQ_ratio_attn_L64_b6, deltaQ_ratio_resn_L64_b6, deltaQ_ratio_tanh_L64_b6, deltaQ_ratio_combined64_L64_b6, deltaQ_ratio_combined_L64_b6]

gamma_L128_b7_ratio_ls = [gamma_ratio_base32_L128_b7, gamma_ratio_attn_L128_b7, gamma_ratio_resn_L128_b7, gamma_ratio_tanh_L128_b7, gamma_ratio_combined64_L128_b7, gamma_ratio_combined_L128_b7]

deltaQ_L128_b7_ratio_ls = [deltaQ_ratio_base32_L128_b7, deltaQ_ratio_attn_L128_b7, deltaQ_ratio_resn_L128_b7, deltaQ_ratio_tanh_L128_b7, deltaQ_ratio_combined64_L128_b7, deltaQ_ratio_combined_L128_b7]


fig, ax = default_plot()

ax.errorbar(np.arange(len(gamma_L32_b6_ratio_ls))-0.05, [gv.mean(gamma_ratio) for gamma_ratio in gamma_L32_b6_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_L32_b6_ratio_ls], label="$\\beta=6$, $V=32^2$", marker="x", **errorb)

ax.errorbar(np.arange(len(gamma_L64_b6_ratio_ls)), [gv.mean(gamma_ratio) for gamma_ratio in gamma_L64_b6_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_L64_b6_ratio_ls], label="$\\beta=6$, $V=64^2$", marker="x", **errorb)

ax.errorbar(np.arange(len(gamma_L128_b7_ratio_ls))+0.05, [gv.mean(gamma_ratio) for gamma_ratio in gamma_L128_b7_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_L128_b7_ratio_ls], label="$\\beta=7$, $V=128^2$", marker="x", **errorb)


ax.set_ylabel('$\\gamma (16)_{\\mathrm{Base}} ~/~ \\gamma (16)$', **fs_p)
ax.set_ylim(0.8, 1.6)
ax.set_xticks(np.arange(len(gamma_L64_b6_ratio_ls)))
ax.set_xticklabels(['Base32', 'Attn', 'Resn', 'Tanh', 'Comb', 'Comb32'], **fs_p)
ax.legend(ncol=2, loc='upper left', **fs_small_p)
plt.tight_layout()
plt.savefig('plots/summary_train_b3_L32_gamma.pdf', transparent=True)
plt.show()


fig, ax = default_plot()

ax.errorbar(np.arange(len(deltaQ_L32_b6_ratio_ls))-0.05, [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_L32_b6_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_L32_b6_ratio_ls], label="$\\beta=6$, $V=32^2$", **errorb_circle)

ax.errorbar(np.arange(len(deltaQ_L64_b6_ratio_ls)), [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_L64_b6_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_L64_b6_ratio_ls], label="$\\beta=6$, $V=64^2$", **errorb_circle)

ax.errorbar(np.arange(len(deltaQ_L128_b7_ratio_ls))+0.05, [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_L128_b7_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_L128_b7_ratio_ls], label="$\\beta=7$, $V=128^2$", **errorb_circle)

ax.set_ylabel('$\\Delta Q ~/~ \\Delta Q_{\\mathrm{Base}}$', **fs_p)
ax.set_ylim(0.8, 1.6)
ax.set_xticks(np.arange(len(deltaQ_L64_b6_ratio_ls)))
ax.set_xticklabels(['Base32', 'Attn', 'Resn', 'Tanh', 'Comb', 'Comb32'], **fs_p)
ax.legend(ncol=2, loc='upper left', **fs_small_p)
plt.tight_layout()
plt.savefig('plots/summary_train_b3_L32_deltaQ.pdf', transparent=True)
plt.show()

# %%
