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
#! hmc b6 L32

hmc_L32_b6_topo = {}
hmc_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    hmc_L32_b6_topo[rand_seed] = np.loadtxt(f'../scaling/dumps/topo_hmc_L32_beta6.0_nsteps{n_steps}_{rand_seed}.csv')
    hmc_L32_b6_deltaQ[rand_seed] = np.mean([ abs(hmc_L32_b6_topo[rand_seed][i] - hmc_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(hmc_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

hmc_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    hmc_L32_b6_auto[rand_seed] = auto_from_chi(hmc_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
hmc_L32_b6_auto_arr = np.array([hmc_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L32_b6_auto_arr, hmc_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
hmc_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
hmc_L32_b6_deltaQ_arr = np.array([hmc_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L32_b6_deltaQ_arr, hmc_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
hmc_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_hmc = 1 / (1 - hmc_L32_b6_auto_avg[16])
gamma_hmc = 1 / (1 - hmc_L32_b6_auto_avg[16])
gamma_ratio_hmc_L32_b6 = gamma_hmc / gamma_hmc
deltaQ_ratio_hmc_L32_b6 = hmc_L32_b6_deltaQ_avg / hmc_L32_b6_deltaQ_avg

print(f"gamma ratio for HMC L32 b6: {gamma_ratio_hmc_L32_b6}")
print(f"deltaQ ratio for HMC L32 b6: {deltaQ_ratio_hmc_L32_b6}")


# %%
#! hmc b6 L64

hmc_L64_b6_topo = {}
hmc_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    hmc_L64_b6_topo[rand_seed] = np.loadtxt(f'../scaling/dumps/topo_hmc_L64_beta6.0_nsteps{n_steps}_{rand_seed}.csv')
    hmc_L64_b6_deltaQ[rand_seed] = np.mean([ abs(hmc_L64_b6_topo[rand_seed][i] - hmc_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(hmc_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

hmc_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    hmc_L64_b6_auto[rand_seed] = auto_from_chi(hmc_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
hmc_L64_b6_auto_arr = np.array([hmc_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L64_b6_auto_arr, hmc_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
hmc_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
hmc_L64_b6_deltaQ_arr = np.array([hmc_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L64_b6_deltaQ_arr, hmc_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
hmc_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]


gamma_hmc = 1 / (1 - hmc_L64_b6_auto_avg[16])
gamma_hmc = 1 / (1 - hmc_L64_b6_auto_avg[16])
gamma_ratio_hmc_L64_b6 = gamma_hmc / gamma_hmc
deltaQ_ratio_hmc_L64_b6 = hmc_L64_b6_deltaQ_avg / hmc_L64_b6_deltaQ_avg

print(f"gamma ratio for HMC L64 b6: {gamma_ratio_hmc_L64_b6}")
print(f"deltaQ ratio for HMC L64 b6: {deltaQ_ratio_hmc_L64_b6}")



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
jk_auto_arr = jackknife( np.concatenate([hmc_L32_b6_auto_arr, fthmc_allp_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_allp_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_allp_L32_b6_deltaQ_arr = np.array([fthmc_allp_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L32_b6_deltaQ_arr, fthmc_allp_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_allp_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_allp_L32_b6_auto_avg[16])
gamma_ratio_allp_L32_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_allp_L32_b6 = fthmc_allp_L32_b6_deltaQ_avg / hmc_L32_b6_deltaQ_avg


print(f"gamma ratio for allp L32 b6: {gamma_ratio_allp_L32_b6}")
print(f"deltaQ ratio for allp L32 b6: {deltaQ_ratio_allp_L32_b6}")


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
jk_auto_arr = jackknife( np.concatenate([hmc_L32_b6_auto_arr, fthmc_allr_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_allr_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_allr_L32_b6_deltaQ_arr = np.array([fthmc_allr_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L32_b6_deltaQ_arr, fthmc_allr_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_allr_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_allr_L32_b6_auto_avg[16])
gamma_ratio_allr_L32_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_allr_L32_b6 = fthmc_allr_L32_b6_deltaQ_avg / hmc_L32_b6_deltaQ_avg


print(f"gamma ratio for allr L32 b6: {gamma_ratio_allr_L32_b6}")
print(f"deltaQ ratio for allr L32 b6: {deltaQ_ratio_allr_L32_b6}")


# %%
#! equal b6 L32

fthmc_equal_L32_b6_topo = {}
fthmc_equal_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_equal_L32_b6_topo[rand_seed] = np.loadtxt(f'../equal_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_equal_train_b3.0_L32_{rand_seed}.csv')
    fthmc_equal_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_equal_L32_b6_topo[rand_seed][i] - fthmc_equal_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_equal_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_equal_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_equal_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_equal_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_equal_L32_b6_auto_arr = np.array([fthmc_equal_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L32_b6_auto_arr, fthmc_equal_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_equal_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_equal_L32_b6_deltaQ_arr = np.array([fthmc_equal_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L32_b6_deltaQ_arr, fthmc_equal_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_equal_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_equal_L32_b6_auto_avg[16])
gamma_ratio_equal_L32_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_equal_L32_b6 = fthmc_equal_L32_b6_deltaQ_avg / hmc_L32_b6_deltaQ_avg


print(f"gamma ratio for equal L32 b6: {gamma_ratio_equal_L32_b6}")
print(f"deltaQ ratio for equal L32 b6: {deltaQ_ratio_equal_L32_b6}")


# %%
#! norect b6 L32

fthmc_norect_L32_b6_topo = {}
fthmc_norect_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_norect_L32_b6_topo[rand_seed] = np.loadtxt(f'../norect_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_norect_train_b3.0_L32_{rand_seed}.csv')
    fthmc_norect_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_norect_L32_b6_topo[rand_seed][i] - fthmc_norect_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_norect_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_norect_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_norect_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_norect_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_norect_L32_b6_auto_arr = np.array([fthmc_norect_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L32_b6_auto_arr, fthmc_norect_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_norect_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_norect_L32_b6_deltaQ_arr = np.array([fthmc_norect_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L32_b6_deltaQ_arr, fthmc_norect_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_norect_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_norect_L32_b6_auto_avg[16])
gamma_ratio_norect_L32_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_norect_L32_b6 = fthmc_norect_L32_b6_deltaQ_avg / hmc_L32_b6_deltaQ_avg


print(f"gamma ratio for norect L32 b6: {gamma_ratio_norect_L32_b6}")
print(f"deltaQ ratio for norect L32 b6: {deltaQ_ratio_norect_L32_b6}")


# %%
#! weight b6 L32

fthmc_weight_L32_b6_topo = {}
fthmc_weight_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_weight_L32_b6_topo[rand_seed] = np.loadtxt(f'../weight_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_weight_train_b3.0_L32_{rand_seed}.csv')
    fthmc_weight_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_weight_L32_b6_topo[rand_seed][i] - fthmc_weight_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_weight_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_weight_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_weight_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_weight_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_weight_L32_b6_auto_arr = np.array([fthmc_weight_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L32_b6_auto_arr, fthmc_weight_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_weight_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_weight_L32_b6_deltaQ_arr = np.array([fthmc_weight_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L32_b6_deltaQ_arr, fthmc_weight_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_weight_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_weight_L32_b6_auto_avg[16])
gamma_ratio_weight_L32_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_weight_L32_b6 = fthmc_weight_L32_b6_deltaQ_avg / hmc_L32_b6_deltaQ_avg


print(f"gamma ratio for weight L32 b6: {gamma_ratio_weight_L32_b6}")
print(f"deltaQ ratio for weight L32 b6: {deltaQ_ratio_weight_L32_b6}")


# %%
#! morecos b6 L32

fthmc_morecos_L32_b6_topo = {}
fthmc_morecos_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_morecos_L32_b6_topo[rand_seed] = np.loadtxt(f'../morecos_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_morecos_train_b3.0_L32_{rand_seed}.csv')
    fthmc_morecos_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_morecos_L32_b6_topo[rand_seed][i] - fthmc_morecos_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_morecos_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_morecos_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_morecos_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_morecos_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_morecos_L32_b6_auto_arr = np.array([fthmc_morecos_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L32_b6_auto_arr, fthmc_morecos_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_morecos_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_morecos_L32_b6_deltaQ_arr = np.array([fthmc_morecos_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L32_b6_deltaQ_arr, fthmc_morecos_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_morecos_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_morecos_L32_b6_auto_avg[16])
gamma_ratio_morecos_L32_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_morecos_L32_b6 = fthmc_morecos_L32_b6_deltaQ_avg / hmc_L32_b6_deltaQ_avg


print(f"gamma ratio for morecos L32 b6: {gamma_ratio_morecos_L32_b6}")
print(f"deltaQ ratio for morecos L32 b6: {deltaQ_ratio_morecos_L32_b6}")


# %%
#! moresin b6 L32

fthmc_moresin_L32_b6_topo = {}
fthmc_moresin_L32_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_moresin_L32_b6_topo[rand_seed] = np.loadtxt(f'../moresin_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps{n_steps}_moresin_train_b3.0_L32_{rand_seed}.csv')
    fthmc_moresin_L32_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_moresin_L32_b6_topo[rand_seed][i] - fthmc_moresin_L32_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_moresin_L32_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 32**2

fthmc_moresin_L32_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_moresin_L32_b6_auto[rand_seed] = auto_from_chi(fthmc_moresin_L32_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_moresin_L32_b6_auto_arr = np.array([fthmc_moresin_L32_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L32_b6_auto_arr, fthmc_moresin_L32_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L32_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_moresin_L32_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_moresin_L32_b6_deltaQ_arr = np.array([fthmc_moresin_L32_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L32_b6_deltaQ_arr, fthmc_moresin_L32_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L32_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_moresin_L32_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L32_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_moresin_L32_b6_auto_avg[16])
gamma_ratio_moresin_L32_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_moresin_L32_b6 = fthmc_moresin_L32_b6_deltaQ_avg / hmc_L32_b6_deltaQ_avg


print(f"gamma ratio for moresin L32 b6: {gamma_ratio_moresin_L32_b6}")
print(f"deltaQ ratio for moresin L32 b6: {deltaQ_ratio_moresin_L32_b6}")


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
jk_auto_arr = jackknife( np.concatenate([hmc_L64_b6_auto_arr, fthmc_allp_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_allp_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_allp_L64_b6_deltaQ_arr = np.array([fthmc_allp_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L64_b6_deltaQ_arr, fthmc_allp_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_allp_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_allp_L64_b6_auto_avg[16])
gamma_ratio_allp_L64_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_allp_L64_b6 = fthmc_allp_L64_b6_deltaQ_avg / hmc_L64_b6_deltaQ_avg


print(f"gamma ratio for allp L64 b6: {gamma_ratio_allp_L64_b6}")
print(f"deltaQ ratio for allp L64 b6: {deltaQ_ratio_allp_L64_b6}")


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
jk_auto_arr = jackknife( np.concatenate([hmc_L64_b6_auto_arr, fthmc_allr_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_allr_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_allr_L64_b6_deltaQ_arr = np.array([fthmc_allr_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L64_b6_deltaQ_arr, fthmc_allr_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_allr_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_allr_L64_b6_auto_avg[16])
gamma_ratio_allr_L64_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_allr_L64_b6 = fthmc_allr_L64_b6_deltaQ_avg / hmc_L64_b6_deltaQ_avg


print(f"gamma ratio for allr L64 b6: {gamma_ratio_allr_L64_b6}")
print(f"deltaQ ratio for allr L64 b6: {deltaQ_ratio_allr_L64_b6}")


# %%
#! equal b6 L64

fthmc_equal_L64_b6_topo = {}
fthmc_equal_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_equal_L64_b6_topo[rand_seed] = np.loadtxt(f'../equal_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_equal_train_b3.0_L32_{rand_seed}.csv')
    fthmc_equal_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_equal_L64_b6_topo[rand_seed][i] - fthmc_equal_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_equal_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_equal_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_equal_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_equal_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_equal_L64_b6_auto_arr = np.array([fthmc_equal_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L64_b6_auto_arr, fthmc_equal_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_equal_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_equal_L64_b6_deltaQ_arr = np.array([fthmc_equal_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L64_b6_deltaQ_arr, fthmc_equal_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_equal_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_equal_L64_b6_auto_avg[16])
gamma_ratio_equal_L64_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_equal_L64_b6 = fthmc_equal_L64_b6_deltaQ_avg / hmc_L64_b6_deltaQ_avg


print(f"gamma ratio for equal L64 b6: {gamma_ratio_equal_L64_b6}")
print(f"deltaQ ratio for equal L64 b6: {deltaQ_ratio_equal_L64_b6}")


# %%
#! norect b6 L64

fthmc_norect_L64_b6_topo = {}
fthmc_norect_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_norect_L64_b6_topo[rand_seed] = np.loadtxt(f'../norect_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_norect_train_b3.0_L32_{rand_seed}.csv')
    fthmc_norect_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_norect_L64_b6_topo[rand_seed][i] - fthmc_norect_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_norect_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_norect_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_norect_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_norect_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_norect_L64_b6_auto_arr = np.array([fthmc_norect_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L64_b6_auto_arr, fthmc_norect_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_norect_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_norect_L64_b6_deltaQ_arr = np.array([fthmc_norect_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L64_b6_deltaQ_arr, fthmc_norect_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_norect_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_norect_L64_b6_auto_avg[16])
gamma_ratio_norect_L64_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_norect_L64_b6 = fthmc_norect_L64_b6_deltaQ_avg / hmc_L64_b6_deltaQ_avg


print(f"gamma ratio for norect L64 b6: {gamma_ratio_norect_L64_b6}")
print(f"deltaQ ratio for norect L64 b6: {deltaQ_ratio_norect_L64_b6}")


# %%
#! weight b6 L64

fthmc_weight_L64_b6_topo = {}
fthmc_weight_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_weight_L64_b6_topo[rand_seed] = np.loadtxt(f'../weight_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_weight_train_b3.0_L32_{rand_seed}.csv')
    fthmc_weight_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_weight_L64_b6_topo[rand_seed][i] - fthmc_weight_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_weight_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_weight_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_weight_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_weight_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_weight_L64_b6_auto_arr = np.array([fthmc_weight_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L64_b6_auto_arr, fthmc_weight_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_weight_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_weight_L64_b6_deltaQ_arr = np.array([fthmc_weight_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L64_b6_deltaQ_arr, fthmc_weight_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_weight_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_weight_L64_b6_auto_avg[16])
gamma_ratio_weight_L64_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_weight_L64_b6 = fthmc_weight_L64_b6_deltaQ_avg / hmc_L64_b6_deltaQ_avg


print(f"gamma ratio for weight L64 b6: {gamma_ratio_weight_L64_b6}")
print(f"deltaQ ratio for weight L64 b6: {deltaQ_ratio_weight_L64_b6}")


# %%
#! morecos b6 L64

fthmc_morecos_L64_b6_topo = {}
fthmc_morecos_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_morecos_L64_b6_topo[rand_seed] = np.loadtxt(f'../morecos_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_morecos_train_b3.0_L32_{rand_seed}.csv')
    fthmc_morecos_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_morecos_L64_b6_topo[rand_seed][i] - fthmc_morecos_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_morecos_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_morecos_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_morecos_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_morecos_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_morecos_L64_b6_auto_arr = np.array([fthmc_morecos_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L64_b6_auto_arr, fthmc_morecos_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_morecos_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_morecos_L64_b6_deltaQ_arr = np.array([fthmc_morecos_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L64_b6_deltaQ_arr, fthmc_morecos_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_morecos_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_morecos_L64_b6_auto_avg[16])
gamma_ratio_morecos_L64_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_morecos_L64_b6 = fthmc_morecos_L64_b6_deltaQ_avg / hmc_L64_b6_deltaQ_avg


print(f"gamma ratio for morecos L64 b6: {gamma_ratio_morecos_L64_b6}")
print(f"deltaQ ratio for morecos L64 b6: {deltaQ_ratio_morecos_L64_b6}")


# %%
#! moresin b6 L64

fthmc_moresin_L64_b6_topo = {}
fthmc_moresin_L64_b6_deltaQ = {}
for rand_seed in rand_seed_ls:
    fthmc_moresin_L64_b6_topo[rand_seed] = np.loadtxt(f'../moresin_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps{n_steps}_moresin_train_b3.0_L32_{rand_seed}.csv')
    fthmc_moresin_L64_b6_deltaQ[rand_seed] = np.mean([ abs(fthmc_moresin_L64_b6_topo[rand_seed][i] - fthmc_moresin_L64_b6_topo[rand_seed][i-1]) for i in range(1, len(fthmc_moresin_L64_b6_topo[rand_seed]))])

beta = 6.0
max_lag = 20
volume = 64**2

fthmc_moresin_L64_b6_auto = {}
for rand_seed in rand_seed_ls:
    fthmc_moresin_L64_b6_auto[rand_seed] = auto_from_chi(fthmc_moresin_L64_b6_topo[rand_seed], max_lag=max_lag, beta=beta, volume=volume)

# * auto
fthmc_moresin_L64_b6_auto_arr = np.array([fthmc_moresin_L64_b6_auto[seed] for seed in rand_seed_ls])  # Shape: (8, max_lag + 1)
jk_auto_arr = jackknife( np.concatenate([hmc_L64_b6_auto_arr, fthmc_moresin_L64_b6_auto_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2 * max_lag + 2)
jk_auto_avg = jk_ls_avg(jk_auto_arr) # shape: (2 * max_lag + 2,)
hmc_L64_b6_auto_avg = jk_auto_avg[:max_lag+1]
fthmc_moresin_L64_b6_auto_avg = jk_auto_avg[max_lag+1:]

# * deltaQ
fthmc_moresin_L64_b6_deltaQ_arr = np.array([fthmc_moresin_L64_b6_deltaQ[seed] for seed in rand_seed_ls]).reshape(-1, 1) # Shape: (8, 1)
jk_deltaQ_arr = jackknife( np.concatenate([hmc_L64_b6_deltaQ_arr, fthmc_moresin_L64_b6_deltaQ_arr], axis=1) ) # concatenate for jk_ls_avg, shape: (8, 2)
jk_deltaQ_avg = jk_ls_avg(jk_deltaQ_arr) # shape: (2,)
hmc_L64_b6_deltaQ_avg = jk_deltaQ_avg[0]
fthmc_moresin_L64_b6_deltaQ_avg = jk_deltaQ_avg[1]

gamma_hmc = 1 / (1 - hmc_L64_b6_auto_avg[16])
gamma_fthmc = 1 / (1 - fthmc_moresin_L64_b6_auto_avg[16])
gamma_ratio_moresin_L64_b6 = gamma_hmc / gamma_fthmc
deltaQ_ratio_moresin_L64_b6 = fthmc_moresin_L64_b6_deltaQ_avg / hmc_L64_b6_deltaQ_avg


print(f"gamma ratio for moresin L64 b6: {gamma_ratio_moresin_L64_b6}")
print(f"deltaQ ratio for moresin L64 b6: {deltaQ_ratio_moresin_L64_b6}")


# %%
#! summary

gamma_L32_b6_ratio_ls = [gamma_ratio_allp_L32_b6, gamma_ratio_allr_L32_b6, gamma_ratio_equal_L32_b6, gamma_ratio_norect_L32_b6, gamma_ratio_weight_L32_b6, gamma_ratio_morecos_L32_b6, gamma_ratio_moresin_L32_b6]

deltaQ_L32_b6_ratio_ls = [deltaQ_ratio_allp_L32_b6, deltaQ_ratio_allr_L32_b6, deltaQ_ratio_equal_L32_b6, deltaQ_ratio_norect_L32_b6, deltaQ_ratio_weight_L32_b6, deltaQ_ratio_morecos_L32_b6, deltaQ_ratio_moresin_L32_b6]

gamma_L64_b6_ratio_ls = [gamma_ratio_allp_L64_b6, gamma_ratio_allr_L64_b6, gamma_ratio_equal_L64_b6, gamma_ratio_norect_L64_b6, gamma_ratio_weight_L64_b6, gamma_ratio_morecos_L64_b6, gamma_ratio_moresin_L64_b6]

deltaQ_L64_b6_ratio_ls = [deltaQ_ratio_allp_L64_b6, deltaQ_ratio_allr_L64_b6, deltaQ_ratio_equal_L64_b6, deltaQ_ratio_norect_L64_b6, deltaQ_ratio_weight_L64_b6, deltaQ_ratio_morecos_L64_b6, deltaQ_ratio_moresin_L64_b6]

fig, ax = default_plot()

ax.errorbar(np.arange(len(gamma_L32_b6_ratio_ls))-0.05, [gv.mean(gamma_ratio) for gamma_ratio in gamma_L32_b6_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_L32_b6_ratio_ls], label="$\\beta=6$, $V=32^2$", marker="x", **errorb)
ax.errorbar(np.arange(len(gamma_L64_b6_ratio_ls))+0.05, [gv.mean(gamma_ratio) for gamma_ratio in gamma_L64_b6_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_L64_b6_ratio_ls], label="$\\beta=6$, $V=64^2$", marker="o", **errorb)

ax.set_ylabel('$\\gamma (16)_{\\mathrm{Base}} ~/~ \\gamma (16)$', **fs_p)
ax.set_ylim(1.5, 4.5)
ax.set_xticks(np.arange(len(gamma_L32_b6_ratio_ls)))
ax.set_xticklabels(['allp', 'allr', 'equal', 'norect', 'weight', 'morecos', 'moresin'], **fs_small_p)
ax.legend(ncol=2, loc='upper right', **fs_small_p)
plt.tight_layout()
plt.savefig('plots/addcos_summary_train_b3_L32_gamma.pdf', transparent=True)
plt.show()


fig, ax = default_plot()

ax.errorbar(np.arange(len(deltaQ_L32_b6_ratio_ls))-0.05, [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_L32_b6_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_L32_b6_ratio_ls], label="$\\beta=6$, $V=32^2$", **errorb_circle)
ax.errorbar(np.arange(len(deltaQ_L64_b6_ratio_ls))+0.05, [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_L64_b6_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_L64_b6_ratio_ls], label="$\\beta=6$, $V=64^2$", **errorb_circle)

ax.set_ylabel('$\\Delta Q ~/~ \\Delta Q_{\\mathrm{Base}}$', **fs_p)
ax.set_ylim(1.5, 4.2)
ax.set_xticks(np.arange(len(deltaQ_L32_b6_ratio_ls)))
ax.set_xticklabels(['allp', 'allr', 'equal', 'norect', 'weight', 'morecos', 'moresin'], **fs_small_p)
ax.legend(ncol=2, loc='upper right', **fs_small_p)
plt.tight_layout()
plt.savefig('plots/addcos_summary_train_b3_L32_deltaQ.pdf', transparent=True)
plt.show()

# %%
