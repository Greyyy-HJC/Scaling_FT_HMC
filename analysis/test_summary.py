# %%
import numpy as np
import sys
import gvar as gv
sys.path.append('/eagle/fthmc/run')
from Scaling_FT_HMC.utils.func import auto_from_chi
from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import jackknife, jk_ls_avg


# %%
#! base b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_base_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_train_b3.0_L32_1029.csv')
fthmc_base_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_train_b3.0_L32_1107.csv')
fthmc_base_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_train_b3.0_L32_1331.csv')
fthmc_base_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_train_b3.0_L32_1984.csv')
fthmc_base_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_train_b3.0_L32_1999.csv')
fthmc_base_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_train_b3.0_L32_2008.csv')
fthmc_base_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_train_b3.0_L32_2017.csv')
fthmc_base_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_base_L32_b6_auto_1029 = auto_from_chi(fthmc_base_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_1107 = auto_from_chi(fthmc_base_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_1331 = auto_from_chi(fthmc_base_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_1984 = auto_from_chi(fthmc_base_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_1999 = auto_from_chi(fthmc_base_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_2008 = auto_from_chi(fthmc_base_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_2017 = auto_from_chi(fthmc_base_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_2025 = auto_from_chi(fthmc_base_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1029 = 1 / (1 - fthmc_base_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_base_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_base_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_base_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_base_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_base_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_base_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_base_L32_b6_auto_2025[16])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

base_L32_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L32_b6_jk)

print("\n>>> base b6 L32 gamma ratio")
print(f"mean(16) for base b6 L32: {gv.mean(base_L32_b6_gamma_ratio)}")
print(f"std(16) for base b6 L32: {gv.sdev(base_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_base_L32_b6_topo_1029[i] - fthmc_base_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_base_L32_b6_topo_1107[i] - fthmc_base_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_base_L32_b6_topo_1331[i] - fthmc_base_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_base_L32_b6_topo_1984[i] - fthmc_base_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_base_L32_b6_topo_1999[i] - fthmc_base_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_base_L32_b6_topo_2008[i] - fthmc_base_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_base_L32_b6_topo_2017[i] - fthmc_base_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_base_L32_b6_topo_2025[i] - fthmc_base_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

base_L32_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L32_b6_jk) / deltaQ_hmc_L32_b6_mean

print("\n>>> base b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for base b6 L32: {gv.mean(base_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for base b6 L32: {gv.sdev(base_L32_b6_deltaQ_ratio)}")

# %%
#! base b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')

fthmc_base_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_train_b3.0_L32_1029.csv')
fthmc_base_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_train_b3.0_L32_1107.csv')
fthmc_base_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_train_b3.0_L32_1331.csv')
fthmc_base_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_train_b3.0_L32_1984.csv')
fthmc_base_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_train_b3.0_L32_1999.csv')
fthmc_base_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_train_b3.0_L32_2008.csv')
fthmc_base_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_train_b3.0_L32_2017.csv')
fthmc_base_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_base_L64_b6_auto_1029 = auto_from_chi(fthmc_base_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_1107 = auto_from_chi(fthmc_base_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_1331 = auto_from_chi(fthmc_base_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_1984 = auto_from_chi(fthmc_base_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_1999 = auto_from_chi(fthmc_base_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_2008 = auto_from_chi(fthmc_base_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_2017 = auto_from_chi(fthmc_base_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_2025 = auto_from_chi(fthmc_base_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_base_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_base_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_base_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_base_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_base_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_base_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_base_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_base_L64_b6_auto_2025[idx])


gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

base_L64_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L64_b6_jk)

print("\n>>> base b6 L64 gamma ratio")
print(f"mean({idx}) for base b6 L64: {gv.mean(base_L64_b6_gamma_ratio)}")
print(f"std({idx}) for base b6 L64: {gv.sdev(base_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]

deltaQ_fthmc_1029 = [ abs(fthmc_base_L64_b6_topo_1029[i] - fthmc_base_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_base_L64_b6_topo_1107[i] - fthmc_base_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_base_L64_b6_topo_1331[i] - fthmc_base_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_base_L64_b6_topo_1984[i] - fthmc_base_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_base_L64_b6_topo_1999[i] - fthmc_base_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_base_L64_b6_topo_2008[i] - fthmc_base_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_base_L64_b6_topo_2017[i] - fthmc_base_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_base_L64_b6_topo_2025[i] - fthmc_base_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

base_L64_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L64_b6_jk) / deltaQ_hmc_L64_b6_mean

print("\n>>> base b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for base b6 L64: {gv.mean(base_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for base b6 L64: {gv.sdev(base_L64_b6_deltaQ_ratio)}")

# %%
#! tanh b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_tanh_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_tanh_train_b3.0_L32_1029.csv')
fthmc_tanh_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_tanh_train_b3.0_L32_1107.csv')
fthmc_tanh_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_tanh_train_b3.0_L32_1331.csv')
fthmc_tanh_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_tanh_train_b3.0_L32_1984.csv')
fthmc_tanh_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_tanh_train_b3.0_L32_1999.csv')
fthmc_tanh_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_tanh_train_b3.0_L32_2008.csv')
fthmc_tanh_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_tanh_train_b3.0_L32_2017.csv')
fthmc_tanh_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_tanh_train_b3.0_L32_2025.csv')

beta = 6.0
max_lag = 64
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_tanh_L32_b6_auto_1029 = auto_from_chi(fthmc_tanh_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L32_b6_auto_1107 = auto_from_chi(fthmc_tanh_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L32_b6_auto_1331 = auto_from_chi(fthmc_tanh_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L32_b6_auto_1984 = auto_from_chi(fthmc_tanh_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L32_b6_auto_1999 = auto_from_chi(fthmc_tanh_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L32_b6_auto_2008 = auto_from_chi(fthmc_tanh_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L32_b6_auto_2017 = auto_from_chi(fthmc_tanh_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L32_b6_auto_2025 = auto_from_chi(fthmc_tanh_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])
gamma_fthmc_1029 = 1 / (1 - fthmc_tanh_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_tanh_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_tanh_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_tanh_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_tanh_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_tanh_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_tanh_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_tanh_L32_b6_auto_2025[16])


gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

tanh_L32_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L32_b6_jk)

print("\n>>> tanh b6 L32 gamma ratio")
print(f"mean(16) for tanh b6 L32: {gv.mean(tanh_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_tanh_L32_b6_topo_1029[i] - fthmc_tanh_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_tanh_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_tanh_L32_b6_topo_1107[i] - fthmc_tanh_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_tanh_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_tanh_L32_b6_topo_1331[i] - fthmc_tanh_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_tanh_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_tanh_L32_b6_topo_1984[i] - fthmc_tanh_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_tanh_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_tanh_L32_b6_topo_1999[i] - fthmc_tanh_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_tanh_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_tanh_L32_b6_topo_2008[i] - fthmc_tanh_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_tanh_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_tanh_L32_b6_topo_2017[i] - fthmc_tanh_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_tanh_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_tanh_L32_b6_topo_2025[i] - fthmc_tanh_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_tanh_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

tanh_L32_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L32_b6_jk) / deltaQ_hmc_L32_b6_mean

print("\n>>> tanh b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for tanh b6 L32: {gv.mean(tanh_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for tanh b6 L32: {gv.sdev(tanh_L32_b6_deltaQ_ratio)}")

# %%
#! tanh b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')

fthmc_tanh_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_tanh_train_b3.0_L32_1029.csv')
fthmc_tanh_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_tanh_train_b3.0_L32_1107.csv')
fthmc_tanh_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_tanh_train_b3.0_L32_1331.csv')
fthmc_tanh_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_tanh_train_b3.0_L32_1984.csv')
fthmc_tanh_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_tanh_train_b3.0_L32_1999.csv')
fthmc_tanh_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_tanh_train_b3.0_L32_2008.csv')
fthmc_tanh_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_tanh_train_b3.0_L32_2017.csv')
fthmc_tanh_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/tanh_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_tanh_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_tanh_L64_b6_auto_1029 = auto_from_chi(fthmc_tanh_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L64_b6_auto_1107 = auto_from_chi(fthmc_tanh_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L64_b6_auto_1331 = auto_from_chi(fthmc_tanh_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L64_b6_auto_1984 = auto_from_chi(fthmc_tanh_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L64_b6_auto_1999 = auto_from_chi(fthmc_tanh_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L64_b6_auto_2008 = auto_from_chi(fthmc_tanh_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L64_b6_auto_2017 = auto_from_chi(fthmc_tanh_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_tanh_L64_b6_auto_2025 = auto_from_chi(fthmc_tanh_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_tanh_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_tanh_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_tanh_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_tanh_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_tanh_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_tanh_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_tanh_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_tanh_L64_b6_auto_2025[idx])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

tanh_L64_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L64_b6_jk)

print("\n>>> tanh b6 L64 gamma ratio")
print(f"mean({idx}) for tanh b6 L64: {gv.mean(tanh_L64_b6_gamma_ratio)}")
print(f"std({idx}) for tanh b6 L64: {gv.sdev(tanh_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_tanh_L64_b6_topo_1029[i] - fthmc_tanh_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_tanh_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_tanh_L64_b6_topo_1107[i] - fthmc_tanh_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_tanh_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_tanh_L64_b6_topo_1331[i] - fthmc_tanh_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_tanh_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_tanh_L64_b6_topo_1984[i] - fthmc_tanh_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_tanh_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_tanh_L64_b6_topo_1999[i] - fthmc_tanh_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_tanh_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_tanh_L64_b6_topo_2008[i] - fthmc_tanh_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_tanh_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_tanh_L64_b6_topo_2017[i] - fthmc_tanh_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_tanh_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_tanh_L64_b6_topo_2025[i] - fthmc_tanh_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_tanh_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

tanh_L64_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L64_b6_jk) / deltaQ_hmc_L64_b6_mean

print("\n>>> tanh b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for tanh b6 L64: {gv.mean(tanh_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for tanh b6 L64: {gv.sdev(tanh_L64_b6_deltaQ_ratio)}")



# %%
#! arctan b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_arctan_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_arctan_train_b3.0_L32_1029.csv')
fthmc_arctan_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_arctan_train_b3.0_L32_1107.csv')
fthmc_arctan_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_arctan_train_b3.0_L32_1331.csv')
fthmc_arctan_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_arctan_train_b3.0_L32_1984.csv')
fthmc_arctan_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_arctan_train_b3.0_L32_1999.csv')
fthmc_arctan_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_arctan_train_b3.0_L32_2008.csv')
fthmc_arctan_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_arctan_train_b3.0_L32_2017.csv')
fthmc_arctan_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_arctan_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_arctan_L32_b6_auto_1029 = auto_from_chi(fthmc_arctan_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_1107 = auto_from_chi(fthmc_arctan_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_1331 = auto_from_chi(fthmc_arctan_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_1984 = auto_from_chi(fthmc_arctan_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_1999 = auto_from_chi(fthmc_arctan_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_2008 = auto_from_chi(fthmc_arctan_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_2017 = auto_from_chi(fthmc_arctan_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_2025 = auto_from_chi(fthmc_arctan_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1029 = 1 / (1 - fthmc_arctan_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_arctan_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_arctan_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_arctan_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_arctan_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_arctan_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_arctan_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_arctan_L32_b6_auto_2025[16])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

arctan_L32_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L32_b6_jk)

print("\n>>> arctan b6 L32 gamma ratio")
print(f"mean(16) for arctan b6 L32: {gv.mean(arctan_L32_b6_gamma_ratio)}")
print(f"std(16) for arctan b6 L32: {gv.sdev(arctan_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]

deltaQ_fthmc_1029 = [ abs(fthmc_arctan_L32_b6_topo_1029[i] - fthmc_arctan_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_arctan_L32_b6_topo_1107[i] - fthmc_arctan_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_arctan_L32_b6_topo_1331[i] - fthmc_arctan_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_arctan_L32_b6_topo_1984[i] - fthmc_arctan_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_arctan_L32_b6_topo_1999[i] - fthmc_arctan_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_arctan_L32_b6_topo_2008[i] - fthmc_arctan_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_arctan_L32_b6_topo_2017[i] - fthmc_arctan_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_arctan_L32_b6_topo_2025[i] - fthmc_arctan_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

arctan_L32_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L32_b6_jk) / deltaQ_hmc_L32_b6_mean

print("\n>>> arctan b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for arctan b6 L32: {gv.mean(arctan_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for arctan b6 L32: {gv.sdev(arctan_L32_b6_deltaQ_ratio)}")

# %%
#! arctan b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')

fthmc_arctan_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_arctan_train_b3.0_L32_1029.csv')
fthmc_arctan_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_arctan_train_b3.0_L32_1107.csv')
fthmc_arctan_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_arctan_train_b3.0_L32_1331.csv')
fthmc_arctan_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_arctan_train_b3.0_L32_1984.csv')
fthmc_arctan_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_arctan_train_b3.0_L32_1999.csv')
fthmc_arctan_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_arctan_train_b3.0_L32_2008.csv')
fthmc_arctan_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_arctan_train_b3.0_L32_2017.csv')
fthmc_arctan_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_arctan_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_arctan_L64_b6_auto_1029 = auto_from_chi(fthmc_arctan_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_1107 = auto_from_chi(fthmc_arctan_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_1331 = auto_from_chi(fthmc_arctan_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_1984 = auto_from_chi(fthmc_arctan_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_1999 = auto_from_chi(fthmc_arctan_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_2008 = auto_from_chi(fthmc_arctan_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_2017 = auto_from_chi(fthmc_arctan_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_2025 = auto_from_chi(fthmc_arctan_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_arctan_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_arctan_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_arctan_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_arctan_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_arctan_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_arctan_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_arctan_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_arctan_L64_b6_auto_2025[idx])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

arctan_L64_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L64_b6_jk)

print("\n>>> arctan b6 L64 gamma ratio")
print(f"mean({idx}) for arctan b6 L64: {gv.mean(arctan_L64_b6_gamma_ratio)}")
print(f"std({idx}) for arctan b6 L64: {gv.sdev(arctan_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]

deltaQ_fthmc_1029 = [ abs(fthmc_arctan_L64_b6_topo_1029[i] - fthmc_arctan_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_arctan_L64_b6_topo_1107[i] - fthmc_arctan_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_arctan_L64_b6_topo_1331[i] - fthmc_arctan_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_arctan_L64_b6_topo_1984[i] - fthmc_arctan_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_arctan_L64_b6_topo_1999[i] - fthmc_arctan_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_arctan_L64_b6_topo_2008[i] - fthmc_arctan_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_arctan_L64_b6_topo_2017[i] - fthmc_arctan_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_arctan_L64_b6_topo_2025[i] - fthmc_arctan_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

arctan_L64_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L64_b6_jk) / deltaQ_hmc_L64_b6_mean

print("\n>>> arctan b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for arctan b6 L64: {gv.mean(arctan_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for arctan b6 L64: {gv.sdev(arctan_L64_b6_deltaQ_ratio)}")


# %%
#! allp b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_allp_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allp_train_b3.0_L32_1029.csv')
fthmc_allp_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allp_train_b3.0_L32_1107.csv')
fthmc_allp_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allp_train_b3.0_L32_1331.csv')
fthmc_allp_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allp_train_b3.0_L32_1984.csv')
fthmc_allp_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allp_train_b3.0_L32_1999.csv')
fthmc_allp_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allp_train_b3.0_L32_2008.csv')
fthmc_allp_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allp_train_b3.0_L32_2017.csv')
fthmc_allp_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allp_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_allp_L32_b6_auto_1029 = auto_from_chi(fthmc_allp_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L32_b6_auto_1107 = auto_from_chi(fthmc_allp_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L32_b6_auto_1331 = auto_from_chi(fthmc_allp_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L32_b6_auto_1984 = auto_from_chi(fthmc_allp_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L32_b6_auto_1999 = auto_from_chi(fthmc_allp_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L32_b6_auto_2008 = auto_from_chi(fthmc_allp_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L32_b6_auto_2017 = auto_from_chi(fthmc_allp_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L32_b6_auto_2025 = auto_from_chi(fthmc_allp_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1029 = 1 / (1 - fthmc_allp_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_allp_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_allp_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_allp_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_allp_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_allp_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_allp_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_allp_L32_b6_auto_2025[16])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

allp_L32_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L32_b6_jk)

print("\n>>> allp b6 L32 gamma ratio")
print(f"mean(16) for allp b6 L32: {gv.mean(allp_L32_b6_gamma_ratio)}")
print(f"std(16) for allp b6 L32: {gv.sdev(allp_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_allp_L32_b6_topo_1029[i] - fthmc_allp_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_allp_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_allp_L32_b6_topo_1107[i] - fthmc_allp_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_allp_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_allp_L32_b6_topo_1331[i] - fthmc_allp_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_allp_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_allp_L32_b6_topo_1984[i] - fthmc_allp_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_allp_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_allp_L32_b6_topo_1999[i] - fthmc_allp_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_allp_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_allp_L32_b6_topo_2008[i] - fthmc_allp_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_allp_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_allp_L32_b6_topo_2017[i] - fthmc_allp_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_allp_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_allp_L32_b6_topo_2025[i] - fthmc_allp_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_allp_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

allp_L32_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L32_b6_jk) / deltaQ_hmc_L32_b6_mean

print("\n>>> allp b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for allp b6 L32: {gv.mean(allp_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for allp b6 L32: {gv.sdev(allp_L32_b6_deltaQ_ratio)}")

# %%
#! allp b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')


fthmc_allp_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allp_train_b3.0_L32_1029.csv')
fthmc_allp_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allp_train_b3.0_L32_1107.csv')
fthmc_allp_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allp_train_b3.0_L32_1331.csv')
fthmc_allp_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allp_train_b3.0_L32_1984.csv')
fthmc_allp_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allp_train_b3.0_L32_1999.csv')
fthmc_allp_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allp_train_b3.0_L32_2008.csv')
fthmc_allp_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allp_train_b3.0_L32_2017.csv')
fthmc_allp_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allp_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allp_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L64_b6_auto_1029 = auto_from_chi(fthmc_allp_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L64_b6_auto_1107 = auto_from_chi(fthmc_allp_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L64_b6_auto_1331 = auto_from_chi(fthmc_allp_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L64_b6_auto_1984 = auto_from_chi(fthmc_allp_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L64_b6_auto_1999 = auto_from_chi(fthmc_allp_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L64_b6_auto_2008 = auto_from_chi(fthmc_allp_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L64_b6_auto_2017 = auto_from_chi(fthmc_allp_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allp_L64_b6_auto_2025 = auto_from_chi(fthmc_allp_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_allp_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_allp_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_allp_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_allp_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_allp_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_allp_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_allp_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_allp_L64_b6_auto_2025[idx])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

allp_L64_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L64_b6_jk)


print("\n>>> allp b6 L64 gamma ratio")
print(f"mean({idx}) for allp b6 L64: {gv.mean(allp_L64_b6_gamma_ratio)}")
print(f"std({idx}) for allp b6 L64: {gv.sdev(allp_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]

deltaQ_fthmc_1029 = [ abs(fthmc_allp_L64_b6_topo_1029[i] - fthmc_allp_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_allp_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_allp_L64_b6_topo_1107[i] - fthmc_allp_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_allp_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_allp_L64_b6_topo_1331[i] - fthmc_allp_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_allp_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_allp_L64_b6_topo_1984[i] - fthmc_allp_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_allp_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_allp_L64_b6_topo_1999[i] - fthmc_allp_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_allp_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_allp_L64_b6_topo_2008[i] - fthmc_allp_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_allp_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_allp_L64_b6_topo_2017[i] - fthmc_allp_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_allp_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_allp_L64_b6_topo_2025[i] - fthmc_allp_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_allp_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

allp_L64_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L64_b6_jk) / deltaQ_hmc_L64_b6_mean

print("\n>>> allp b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for allp b6 L64: {gv.mean(allp_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for allp b6 L64: {gv.sdev(allp_L64_b6_deltaQ_ratio)}")

# %%
#! allr b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_allr_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allr_train_b3.0_L32_1029.csv')
fthmc_allr_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allr_train_b3.0_L32_1107.csv')
fthmc_allr_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allr_train_b3.0_L32_1331.csv')
fthmc_allr_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allr_train_b3.0_L32_1984.csv')
fthmc_allr_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allr_train_b3.0_L32_1999.csv')
fthmc_allr_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allr_train_b3.0_L32_2008.csv')
fthmc_allr_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allr_train_b3.0_L32_2017.csv')
fthmc_allr_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_allr_train_b3.0_L32_2025.csv')

beta = 6.0
max_lag = 64
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_allr_L32_b6_auto_1029 = auto_from_chi(fthmc_allr_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L32_b6_auto_1107 = auto_from_chi(fthmc_allr_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L32_b6_auto_1331 = auto_from_chi(fthmc_allr_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L32_b6_auto_1984 = auto_from_chi(fthmc_allr_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L32_b6_auto_1999 = auto_from_chi(fthmc_allr_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L32_b6_auto_2008 = auto_from_chi(fthmc_allr_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L32_b6_auto_2017 = auto_from_chi(fthmc_allr_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L32_b6_auto_2025 = auto_from_chi(fthmc_allr_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)



gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1029 = 1 / (1 - fthmc_allr_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_allr_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_allr_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_allr_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_allr_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_allr_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_allr_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_allr_L32_b6_auto_2025[16])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

allr_L32_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L32_b6_jk)

print("\n>>> allr b6 L32 gamma ratio")
print(f"mean(16) for allr b6 L32: {gv.mean(allr_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_allr_L32_b6_topo_1029[i] - fthmc_allr_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_allr_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_allr_L32_b6_topo_1107[i] - fthmc_allr_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_allr_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_allr_L32_b6_topo_1331[i] - fthmc_allr_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_allr_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_allr_L32_b6_topo_1984[i] - fthmc_allr_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_allr_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_allr_L32_b6_topo_1999[i] - fthmc_allr_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_allr_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_allr_L32_b6_topo_2008[i] - fthmc_allr_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_allr_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_allr_L32_b6_topo_2017[i] - fthmc_allr_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_allr_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_allr_L32_b6_topo_2025[i] - fthmc_allr_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_allr_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

allr_L32_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L32_b6_jk) / deltaQ_hmc_L32_b6_mean

print("\n>>> allr b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for allr b6 L32: {gv.mean(allr_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for allr b6 L32: {gv.sdev(allr_L32_b6_deltaQ_ratio)}")

# %%
#! allr b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')

fthmc_allr_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allr_train_b3.0_L32_1029.csv')
fthmc_allr_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allr_train_b3.0_L32_1107.csv')
fthmc_allr_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allr_train_b3.0_L32_1331.csv')
fthmc_allr_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allr_train_b3.0_L32_1984.csv')
fthmc_allr_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allr_train_b3.0_L32_1999.csv')
fthmc_allr_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allr_train_b3.0_L32_2008.csv')
fthmc_allr_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allr_train_b3.0_L32_2017.csv')
fthmc_allr_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/allr_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_allr_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_allr_L64_b6_auto_1029 = auto_from_chi(fthmc_allr_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L64_b6_auto_1107 = auto_from_chi(fthmc_allr_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L64_b6_auto_1331 = auto_from_chi(fthmc_allr_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L64_b6_auto_1984 = auto_from_chi(fthmc_allr_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L64_b6_auto_1999 = auto_from_chi(fthmc_allr_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L64_b6_auto_2008 = auto_from_chi(fthmc_allr_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L64_b6_auto_2017 = auto_from_chi(fthmc_allr_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_allr_L64_b6_auto_2025 = auto_from_chi(fthmc_allr_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_allr_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_allr_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_allr_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_allr_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_allr_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_allr_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_allr_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_allr_L64_b6_auto_2025[idx])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

allr_L64_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L64_b6_jk)

print("\n>>> allr b6 L64 gamma ratio")
print(f"mean({idx}) for allr b6 L64: {gv.mean(allr_L64_b6_gamma_ratio)}")
print(f"std({idx}) for allr b6 L64: {gv.sdev(allr_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_allr_L64_b6_topo_1029[i] - fthmc_allr_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_allr_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_allr_L64_b6_topo_1107[i] - fthmc_allr_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_allr_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_allr_L64_b6_topo_1331[i] - fthmc_allr_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_allr_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_allr_L64_b6_topo_1984[i] - fthmc_allr_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_allr_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_allr_L64_b6_topo_1999[i] - fthmc_allr_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_allr_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_allr_L64_b6_topo_2008[i] - fthmc_allr_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_allr_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_allr_L64_b6_topo_2017[i] - fthmc_allr_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_allr_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_allr_L64_b6_topo_2025[i] - fthmc_allr_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_allr_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

allr_L64_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L64_b6_jk) / deltaQ_hmc_L64_b6_mean

print("\n>>> allr b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for allr b6 L64: {gv.mean(allr_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for allr b6 L64: {gv.sdev(allr_L64_b6_deltaQ_ratio)}")


# %%
#! 2plaq b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_2plaq_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_2plaq_train_b3.0_L32_1029.csv')
fthmc_2plaq_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_2plaq_train_b3.0_L32_1107.csv')
fthmc_2plaq_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_2plaq_train_b3.0_L32_1331.csv')
fthmc_2plaq_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_2plaq_train_b3.0_L32_1984.csv')
fthmc_2plaq_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_2plaq_train_b3.0_L32_1999.csv')
fthmc_2plaq_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_2plaq_train_b3.0_L32_2008.csv')
fthmc_2plaq_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_2plaq_train_b3.0_L32_2017.csv')
fthmc_2plaq_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_2plaq_train_b3.0_L32_2025.csv')

beta = 6.0
max_lag = 64
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_2plaq_L32_b6_auto_1029 = auto_from_chi(fthmc_2plaq_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L32_b6_auto_1107 = auto_from_chi(fthmc_2plaq_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L32_b6_auto_1331 = auto_from_chi(fthmc_2plaq_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L32_b6_auto_1984 = auto_from_chi(fthmc_2plaq_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L32_b6_auto_1999 = auto_from_chi(fthmc_2plaq_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L32_b6_auto_2008 = auto_from_chi(fthmc_2plaq_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L32_b6_auto_2017 = auto_from_chi(fthmc_2plaq_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L32_b6_auto_2025 = auto_from_chi(fthmc_2plaq_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])
gamma_fthmc_1029 = 1 / (1 - fthmc_2plaq_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_2plaq_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_2plaq_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_2plaq_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_2plaq_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_2plaq_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_2plaq_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_2plaq_L32_b6_auto_2025[16])


gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

plaq2_L32_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L32_b6_jk)

print("\n>>> 2plaq b6 L32 gamma ratio")
print(f"mean(16) for 2plaq b6 L32: {gv.mean(plaq2_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_2plaq_L32_b6_topo_1029[i] - fthmc_2plaq_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_2plaq_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_2plaq_L32_b6_topo_1107[i] - fthmc_2plaq_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_2plaq_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_2plaq_L32_b6_topo_1331[i] - fthmc_2plaq_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_2plaq_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_2plaq_L32_b6_topo_1984[i] - fthmc_2plaq_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_2plaq_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_2plaq_L32_b6_topo_1999[i] - fthmc_2plaq_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_2plaq_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_2plaq_L32_b6_topo_2008[i] - fthmc_2plaq_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_2plaq_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_2plaq_L32_b6_topo_2017[i] - fthmc_2plaq_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_2plaq_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_2plaq_L32_b6_topo_2025[i] - fthmc_2plaq_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_2plaq_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

plaq2_L32_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L32_b6_jk) / deltaQ_hmc_L32_b6_mean

print("\n>>> 2plaq b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for 2plaq b6 L32: {gv.mean(plaq2_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for 2plaq b6 L32: {gv.sdev(plaq2_L32_b6_deltaQ_ratio)}")

# %%
#! 2plaq b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')

fthmc_2plaq_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_2plaq_train_b3.0_L32_1029.csv')
fthmc_2plaq_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_2plaq_train_b3.0_L32_1107.csv')
fthmc_2plaq_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_2plaq_train_b3.0_L32_1331.csv')
fthmc_2plaq_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_2plaq_train_b3.0_L32_1984.csv')
fthmc_2plaq_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_2plaq_train_b3.0_L32_1999.csv')
fthmc_2plaq_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_2plaq_train_b3.0_L32_2008.csv')
fthmc_2plaq_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_2plaq_train_b3.0_L32_2017.csv')
fthmc_2plaq_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/2plaq_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_2plaq_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_2plaq_L64_b6_auto_1029 = auto_from_chi(fthmc_2plaq_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L64_b6_auto_1107 = auto_from_chi(fthmc_2plaq_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L64_b6_auto_1331 = auto_from_chi(fthmc_2plaq_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L64_b6_auto_1984 = auto_from_chi(fthmc_2plaq_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L64_b6_auto_1999 = auto_from_chi(fthmc_2plaq_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L64_b6_auto_2008 = auto_from_chi(fthmc_2plaq_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L64_b6_auto_2017 = auto_from_chi(fthmc_2plaq_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_2plaq_L64_b6_auto_2025 = auto_from_chi(fthmc_2plaq_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_2plaq_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_2plaq_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_2plaq_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_2plaq_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_2plaq_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_2plaq_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_2plaq_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_2plaq_L64_b6_auto_2025[idx])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_jk = jackknife([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

plaq2_L64_b6_gamma_ratio = jk_ls_avg(gamma_ratio_L64_b6_jk)

print("\n>>> 2plaq b6 L64 gamma ratio")
print(f"mean({idx}) for 2plaq b6 L64: {gv.mean(plaq2_L64_b6_gamma_ratio)}")
print(f"std({idx}) for 2plaq b6 L64: {gv.sdev(plaq2_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_2plaq_L64_b6_topo_1029[i] - fthmc_2plaq_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_2plaq_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_2plaq_L64_b6_topo_1107[i] - fthmc_2plaq_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_2plaq_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_2plaq_L64_b6_topo_1331[i] - fthmc_2plaq_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_2plaq_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_2plaq_L64_b6_topo_1984[i] - fthmc_2plaq_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_2plaq_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_2plaq_L64_b6_topo_1999[i] - fthmc_2plaq_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_2plaq_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_2plaq_L64_b6_topo_2008[i] - fthmc_2plaq_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_2plaq_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_2plaq_L64_b6_topo_2017[i] - fthmc_2plaq_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_2plaq_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_2plaq_L64_b6_topo_2025[i] - fthmc_2plaq_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_2plaq_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_jk = jackknife([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

plaq2_L64_b6_deltaQ_ratio = jk_ls_avg(deltaQ_fthmc_L64_b6_jk) / deltaQ_hmc_L64_b6_mean

print("\n>>> 2plaq b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for 2plaq b6 L64: {gv.mean(plaq2_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for 2plaq b6 L64: {gv.sdev(plaq2_L64_b6_deltaQ_ratio)}")




# %%
#! summary

gamma_L32_b6_ratio_ls = [base_L32_b6_gamma_ratio, tanh_L32_b6_gamma_ratio, arctan_L32_b6_gamma_ratio, allp_L32_b6_gamma_ratio, allr_L32_b6_gamma_ratio, plaq2_L32_b6_gamma_ratio]

deltaQ_L32_b6_ratio_ls = [base_L32_b6_deltaQ_ratio, tanh_L32_b6_deltaQ_ratio, arctan_L32_b6_deltaQ_ratio, allp_L32_b6_deltaQ_ratio, allr_L32_b6_deltaQ_ratio, plaq2_L32_b6_deltaQ_ratio]

gamma_L64_b6_ratio_ls = [base_L64_b6_gamma_ratio, tanh_L64_b6_gamma_ratio, arctan_L64_b6_gamma_ratio, allp_L64_b6_gamma_ratio, allr_L64_b6_gamma_ratio, plaq2_L64_b6_gamma_ratio]

deltaQ_L64_b6_ratio_ls = [base_L64_b6_deltaQ_ratio, tanh_L64_b6_deltaQ_ratio, arctan_L64_b6_deltaQ_ratio, allp_L64_b6_deltaQ_ratio, allr_L64_b6_deltaQ_ratio, plaq2_L64_b6_deltaQ_ratio]

fig, (ax1, ax2) = default_sub_plot()
# Adjust subplot spacing
plt.subplots_adjust(left=0.1,    # Increase left margin
                    right=0.95,    # Decrease right margin
                    bottom=0.15,   # Increase bottom margin 
                    top=0.95)      # Decrease top margin

ax1.errorbar(np.arange(len(gamma_L32_b6_ratio_ls)), [gv.mean(gamma_ratio) for gamma_ratio in gamma_L32_b6_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_L32_b6_ratio_ls], label="$\\beta=6$, $L=32$", **errorb)
ax2.errorbar(np.arange(len(deltaQ_L32_b6_ratio_ls)), [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_L32_b6_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_L32_b6_ratio_ls], **errorb_circle)

ax1.errorbar(np.arange(len(gamma_L64_b6_ratio_ls)), [gv.mean(gamma_ratio) for gamma_ratio in gamma_L64_b6_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_L64_b6_ratio_ls], label="$\\beta=6$, $L=64$", **errorb)
ax2.errorbar(np.arange(len(deltaQ_L64_b6_ratio_ls)), [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_L64_b6_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_L64_b6_ratio_ls], **errorb_circle)

ax1.set_ylabel('$R_{\\gamma (\\delta =16)}$', **fs_p)
# ax1.set_ylim(1.7, 3.8)
# ax2.set_xlabel('Model', **fs_p)
ax2.set_ylabel('$R_{\\Delta Q}$', **fs_p)
# ax2.set_ylim(1.3, 3.2)
ax2.set_xticks(np.arange(len(gamma_L64_b6_ratio_ls)))
ax2.set_xticklabels(['Base', 'Tanh', 'arctan', 'allp', 'allr', 'plaq2'], fontsize=19)
ax1.legend(ncol=2, loc='upper right', **fs_small_p)
# plt.tight_layout()
plt.savefig('plots/test_summary_train_b3_L32.pdf', transparent=True)
plt.show()
# %%
