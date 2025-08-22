# %%
import numpy as np
import sys
import gvar as gv
sys.path.append('/eagle/fthmc/run')
from Scaling_FT_HMC.utils.func import auto_from_chi
from lametlat.utils.plot_settings import *


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

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

base_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

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
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

base_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

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

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025]) 
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

base_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)


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
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

base_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> base b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for base b6 L64: {gv.mean(base_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for base b6 L64: {gv.sdev(base_L64_b6_deltaQ_ratio)}")


# %%
#! base32 b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_base32_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1029.csv')
fthmc_base32_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1107.csv')
fthmc_base32_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1331.csv')
fthmc_base32_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1984.csv')
fthmc_base32_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1999.csv')
fthmc_base32_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_batch32_train_b3.0_L32_2008.csv')
fthmc_base32_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_batch32_train_b3.0_L32_2017.csv')
fthmc_base32_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_base_batch32_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_base32_L32_b6_auto_1029 = auto_from_chi(fthmc_base32_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L32_b6_auto_1107 = auto_from_chi(fthmc_base32_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L32_b6_auto_1331 = auto_from_chi(fthmc_base32_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L32_b6_auto_1984 = auto_from_chi(fthmc_base32_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L32_b6_auto_1999 = auto_from_chi(fthmc_base32_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L32_b6_auto_2008 = auto_from_chi(fthmc_base32_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L32_b6_auto_2017 = auto_from_chi(fthmc_base32_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L32_b6_auto_2025 = auto_from_chi(fthmc_base32_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1029 = 1 / (1 - fthmc_base32_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_base32_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_base32_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_base32_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_base32_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_base32_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_base32_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_base32_L32_b6_auto_2025[16])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

base32_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> base b6 L32 gamma ratio")
print(f"mean(16) for base b6 L32: {gv.mean(base32_L32_b6_gamma_ratio)}")
print(f"std(16) for base b6 L32: {gv.sdev(base32_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]

deltaQ_fthmc_1029 = [ abs(fthmc_base32_L32_b6_topo_1029[i] - fthmc_base32_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_base32_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_base32_L32_b6_topo_1107[i] - fthmc_base32_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_base32_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_base32_L32_b6_topo_1331[i] - fthmc_base32_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_base32_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_base32_L32_b6_topo_1984[i] - fthmc_base32_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_base32_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_base32_L32_b6_topo_1999[i] - fthmc_base32_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_base32_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_base32_L32_b6_topo_2008[i] - fthmc_base32_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_base32_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_base32_L32_b6_topo_2017[i] - fthmc_base32_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_base32_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_base32_L32_b6_topo_2025[i] - fthmc_base32_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_base32_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

base32_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> base b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for base b6 L32: {gv.mean(base32_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for base b6 L32: {gv.sdev(base32_L32_b6_deltaQ_ratio)}")

# %%
#! base32 b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')

fthmc_base32_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1029.csv')
fthmc_base32_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1107.csv')
fthmc_base32_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1331.csv')
fthmc_base32_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1984.csv')
fthmc_base32_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_batch32_train_b3.0_L32_1999.csv')
fthmc_base32_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_batch32_train_b3.0_L32_2008.csv')
fthmc_base32_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_batch32_train_b3.0_L32_2017.csv')
fthmc_base32_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/base_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_base_batch32_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_base32_L64_b6_auto_1029 = auto_from_chi(fthmc_base32_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L64_b6_auto_1107 = auto_from_chi(fthmc_base32_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L64_b6_auto_1331 = auto_from_chi(fthmc_base32_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L64_b6_auto_1984 = auto_from_chi(fthmc_base32_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L64_b6_auto_1999 = auto_from_chi(fthmc_base32_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L64_b6_auto_2008 = auto_from_chi(fthmc_base32_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L64_b6_auto_2017 = auto_from_chi(fthmc_base32_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base32_L64_b6_auto_2025 = auto_from_chi(fthmc_base32_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_base32_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_base32_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_base32_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_base32_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_base32_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_base32_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_base32_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_base32_L64_b6_auto_2025[idx])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025]) 
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

base32_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)


print("\n>>> base b6 L64 gamma ratio")
print(f"mean({idx}) for base b6 L64: {gv.mean(base32_L64_b6_gamma_ratio)}")
print(f"std({idx}) for base b6 L64: {gv.sdev(base32_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]

deltaQ_fthmc_1029 = [ abs(fthmc_base32_L64_b6_topo_1029[i] - fthmc_base32_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_base32_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_base32_L64_b6_topo_1107[i] - fthmc_base32_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_base32_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_base32_L64_b6_topo_1331[i] - fthmc_base32_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_base32_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_base32_L64_b6_topo_1984[i] - fthmc_base32_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_base32_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_base32_L64_b6_topo_1999[i] - fthmc_base32_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_base32_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_base32_L64_b6_topo_2008[i] - fthmc_base32_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_base32_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_base32_L64_b6_topo_2017[i] - fthmc_base32_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_base32_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_base32_L64_b6_topo_2025[i] - fthmc_base32_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_base32_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

base32_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> base b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for base b6 L64: {gv.mean(base32_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for base b6 L64: {gv.sdev(base32_L64_b6_deltaQ_ratio)}")


# %%
#! attn b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_attn_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_attn_train_b3.0_L32_1029.csv')
fthmc_attn_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_attn_train_b3.0_L32_1107.csv')
fthmc_attn_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_attn_train_b3.0_L32_1331.csv')
fthmc_attn_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_attn_train_b3.0_L32_1984.csv')
fthmc_attn_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_attn_train_b3.0_L32_1999.csv')
fthmc_attn_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_attn_train_b3.0_L32_2008.csv')
fthmc_attn_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_attn_train_b3.0_L32_2017.csv')
fthmc_attn_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_attn_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_attn_L32_b6_auto_1029 = auto_from_chi(fthmc_attn_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_1107 = auto_from_chi(fthmc_attn_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_1331 = auto_from_chi(fthmc_attn_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_1984 = auto_from_chi(fthmc_attn_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_1999 = auto_from_chi(fthmc_attn_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_2008 = auto_from_chi(fthmc_attn_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_2017 = auto_from_chi(fthmc_attn_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_2025 = auto_from_chi(fthmc_attn_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1029 = 1 / (1 - fthmc_attn_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_attn_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_attn_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_attn_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_attn_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_attn_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_attn_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_attn_L32_b6_auto_2025[16])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

attn_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> attn b6 L32 gamma ratio")
print(f"mean(16) for attn b6 L32: {gv.mean(attn_L32_b6_gamma_ratio)}")
print(f"std(16) for attn b6 L32: {gv.sdev(attn_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_attn_L32_b6_topo_1029[i] - fthmc_attn_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_attn_L32_b6_topo_1107[i] - fthmc_attn_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_attn_L32_b6_topo_1331[i] - fthmc_attn_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_attn_L32_b6_topo_1984[i] - fthmc_attn_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_attn_L32_b6_topo_1999[i] - fthmc_attn_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_attn_L32_b6_topo_2008[i] - fthmc_attn_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_attn_L32_b6_topo_2017[i] - fthmc_attn_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_attn_L32_b6_topo_2025[i] - fthmc_attn_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

attn_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> attn b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for attn b6 L32: {gv.mean(attn_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for attn b6 L32: {gv.sdev(attn_L32_b6_deltaQ_ratio)}")

# %%
#! attn b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')


fthmc_attn_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_attn_train_b3.0_L32_1029.csv')
fthmc_attn_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_attn_train_b3.0_L32_1107.csv')
fthmc_attn_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_attn_train_b3.0_L32_1331.csv')
fthmc_attn_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_attn_train_b3.0_L32_1984.csv')
fthmc_attn_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_attn_train_b3.0_L32_1999.csv')
fthmc_attn_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_attn_train_b3.0_L32_2008.csv')
fthmc_attn_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_attn_train_b3.0_L32_2017.csv')
fthmc_attn_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_attn_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_1029 = auto_from_chi(fthmc_attn_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_1107 = auto_from_chi(fthmc_attn_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_1331 = auto_from_chi(fthmc_attn_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_1984 = auto_from_chi(fthmc_attn_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_1999 = auto_from_chi(fthmc_attn_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_2008 = auto_from_chi(fthmc_attn_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_2017 = auto_from_chi(fthmc_attn_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_2025 = auto_from_chi(fthmc_attn_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_attn_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_attn_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_attn_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_attn_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_attn_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_attn_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_attn_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_attn_L64_b6_auto_2025[idx])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

attn_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)


print("\n>>> attn b6 L64 gamma ratio")
print(f"mean({idx}) for attn b6 L64: {gv.mean(attn_L64_b6_gamma_ratio)}")
print(f"std({idx}) for attn b6 L64: {gv.sdev(attn_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]

deltaQ_fthmc_1029 = [ abs(fthmc_attn_L64_b6_topo_1029[i] - fthmc_attn_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_attn_L64_b6_topo_1107[i] - fthmc_attn_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_attn_L64_b6_topo_1331[i] - fthmc_attn_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_attn_L64_b6_topo_1984[i] - fthmc_attn_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_attn_L64_b6_topo_1999[i] - fthmc_attn_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_attn_L64_b6_topo_2008[i] - fthmc_attn_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_attn_L64_b6_topo_2017[i] - fthmc_attn_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_attn_L64_b6_topo_2025[i] - fthmc_attn_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])


attn_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> attn b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for attn b6 L64: {gv.mean(attn_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for attn b6 L64: {gv.sdev(attn_L64_b6_deltaQ_ratio)}")

# %%
#! resn b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_resn_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_resn_train_b3.0_L32_1029.csv')
fthmc_resn_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_resn_train_b3.0_L32_1107.csv')
fthmc_resn_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_resn_train_b3.0_L32_1331.csv')
fthmc_resn_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_resn_train_b3.0_L32_1984.csv')
fthmc_resn_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_resn_train_b3.0_L32_1999.csv')
fthmc_resn_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_resn_train_b3.0_L32_2008.csv')
fthmc_resn_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_resn_train_b3.0_L32_2017.csv')
fthmc_resn_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_resn_train_b3.0_L32_2025.csv')

beta = 6.0
max_lag = 64
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_resn_L32_b6_auto_1029 = auto_from_chi(fthmc_resn_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_1107 = auto_from_chi(fthmc_resn_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_1331 = auto_from_chi(fthmc_resn_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_1984 = auto_from_chi(fthmc_resn_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_1999 = auto_from_chi(fthmc_resn_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_2008 = auto_from_chi(fthmc_resn_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_2017 = auto_from_chi(fthmc_resn_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_2025 = auto_from_chi(fthmc_resn_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)



gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1029 = 1 / (1 - fthmc_resn_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_resn_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_resn_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_resn_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_resn_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_resn_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_resn_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_resn_L32_b6_auto_2025[16])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

resn_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> resn b6 L32 gamma ratio")
print(f"mean(16) for resn b6 L32: {gv.mean(resn_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_resn_L32_b6_topo_1029[i] - fthmc_resn_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_resn_L32_b6_topo_1107[i] - fthmc_resn_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_resn_L32_b6_topo_1331[i] - fthmc_resn_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_resn_L32_b6_topo_1984[i] - fthmc_resn_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_resn_L32_b6_topo_1999[i] - fthmc_resn_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_resn_L32_b6_topo_2008[i] - fthmc_resn_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_resn_L32_b6_topo_2017[i] - fthmc_resn_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_resn_L32_b6_topo_2025[i] - fthmc_resn_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

resn_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> resn b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for resn b6 L32: {gv.mean(resn_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for resn b6 L32: {gv.sdev(resn_L32_b6_deltaQ_ratio)}")

# %%
#! resn b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')

fthmc_resn_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_resn_train_b3.0_L32_1029.csv')
fthmc_resn_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_resn_train_b3.0_L32_1107.csv')
fthmc_resn_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_resn_train_b3.0_L32_1331.csv')
fthmc_resn_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_resn_train_b3.0_L32_1984.csv')
fthmc_resn_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_resn_train_b3.0_L32_1999.csv')
fthmc_resn_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_resn_train_b3.0_L32_2008.csv')
fthmc_resn_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_resn_train_b3.0_L32_2017.csv')
fthmc_resn_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_resn_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_resn_L64_b6_auto_1029 = auto_from_chi(fthmc_resn_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_1107 = auto_from_chi(fthmc_resn_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_1331 = auto_from_chi(fthmc_resn_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_1984 = auto_from_chi(fthmc_resn_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_1999 = auto_from_chi(fthmc_resn_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_2008 = auto_from_chi(fthmc_resn_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_2017 = auto_from_chi(fthmc_resn_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_2025 = auto_from_chi(fthmc_resn_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_resn_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_resn_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_resn_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_resn_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_resn_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_resn_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_resn_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_resn_L64_b6_auto_2025[idx])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

resn_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)

print("\n>>> resn b6 L64 gamma ratio")
print(f"mean({idx}) for resn b6 L64: {gv.mean(resn_L64_b6_gamma_ratio)}")
print(f"std({idx}) for resn b6 L64: {gv.sdev(resn_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_resn_L64_b6_topo_1029[i] - fthmc_resn_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_resn_L64_b6_topo_1107[i] - fthmc_resn_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_resn_L64_b6_topo_1331[i] - fthmc_resn_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_resn_L64_b6_topo_1984[i] - fthmc_resn_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_resn_L64_b6_topo_1999[i] - fthmc_resn_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_resn_L64_b6_topo_2008[i] - fthmc_resn_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_resn_L64_b6_topo_2017[i] - fthmc_resn_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_resn_L64_b6_topo_2025[i] - fthmc_resn_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

resn_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> resn b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for resn b6 L64: {gv.mean(resn_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for resn b6 L64: {gv.sdev(resn_L64_b6_deltaQ_ratio)}")


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

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

tanh_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

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
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

tanh_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

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

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

tanh_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)

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
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

tanh_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> tanh b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for tanh b6 L64: {gv.mean(tanh_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for tanh b6 L64: {gv.sdev(tanh_L64_b6_deltaQ_ratio)}")


# %%
#! combined b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0_nsteps10.csv')

fthmc_combined_L32_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_combined_train_b3.0_L32_1029.csv')
fthmc_combined_L32_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_combined_train_b3.0_L32_1107.csv')
fthmc_combined_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_combined_train_b3.0_L32_1331.csv')
fthmc_combined_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_combined_train_b3.0_L32_1984.csv')
fthmc_combined_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_combined_train_b3.0_L32_1999.csv')
fthmc_combined_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_combined_train_b3.0_L32_2008.csv')
fthmc_combined_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_combined_train_b3.0_L32_2017.csv')
fthmc_combined_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L32_beta6.0_nsteps10_combined_train_b3.0_L32_2025.csv')

beta = 6.0
max_lag = 64
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_combined_L32_b6_auto_1029 = auto_from_chi(fthmc_combined_L32_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L32_b6_auto_1107 = auto_from_chi(fthmc_combined_L32_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L32_b6_auto_1331 = auto_from_chi(fthmc_combined_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L32_b6_auto_1984 = auto_from_chi(fthmc_combined_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L32_b6_auto_1999 = auto_from_chi(fthmc_combined_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L32_b6_auto_2008 = auto_from_chi(fthmc_combined_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L32_b6_auto_2017 = auto_from_chi(fthmc_combined_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L32_b6_auto_2025 = auto_from_chi(fthmc_combined_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)



gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1029 = 1 / (1 - fthmc_combined_L32_b6_auto_1029[16])
gamma_fthmc_1107 = 1 / (1 - fthmc_combined_L32_b6_auto_1107[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_combined_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_combined_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_combined_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_combined_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_combined_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_combined_L32_b6_auto_2025[16])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

combined_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> combined b6 L32 gamma ratio")
print(f"mean(16) for combined b6 L32: {gv.mean(combined_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_combined_L32_b6_topo_1029[i] - fthmc_combined_L32_b6_topo_1029[i-1]) for i in range(1, len(fthmc_combined_L32_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_combined_L32_b6_topo_1107[i] - fthmc_combined_L32_b6_topo_1107[i-1]) for i in range(1, len(fthmc_combined_L32_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_combined_L32_b6_topo_1331[i] - fthmc_combined_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_combined_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_combined_L32_b6_topo_1984[i] - fthmc_combined_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_combined_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_combined_L32_b6_topo_1999[i] - fthmc_combined_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_combined_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_combined_L32_b6_topo_2008[i] - fthmc_combined_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_combined_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_combined_L32_b6_topo_2017[i] - fthmc_combined_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_combined_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_combined_L32_b6_topo_2025[i] - fthmc_combined_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_combined_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

combined_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> combined b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for combined b6 L32: {gv.mean(combined_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for combined b6 L32: {gv.sdev(combined_L32_b6_deltaQ_ratio)}")

# %%
#! combined b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0_nsteps10.csv')

fthmc_combined_L64_b6_topo_1029 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_combined_train_b3.0_L32_1029.csv')
fthmc_combined_L64_b6_topo_1107 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_combined_train_b3.0_L32_1107.csv')
fthmc_combined_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_combined_train_b3.0_L32_1331.csv')
fthmc_combined_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_combined_train_b3.0_L32_1984.csv')
fthmc_combined_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_combined_train_b3.0_L32_1999.csv')
fthmc_combined_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_combined_train_b3.0_L32_2008.csv')
fthmc_combined_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_combined_train_b3.0_L32_2017.csv')
fthmc_combined_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/combined_evaluation/dumps/topo_fthmc_L64_beta6.0_nsteps10_combined_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_combined_L64_b6_auto_1029 = auto_from_chi(fthmc_combined_L64_b6_topo_1029, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L64_b6_auto_1107 = auto_from_chi(fthmc_combined_L64_b6_topo_1107, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L64_b6_auto_1331 = auto_from_chi(fthmc_combined_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L64_b6_auto_1984 = auto_from_chi(fthmc_combined_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L64_b6_auto_1999 = auto_from_chi(fthmc_combined_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L64_b6_auto_2008 = auto_from_chi(fthmc_combined_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L64_b6_auto_2017 = auto_from_chi(fthmc_combined_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_combined_L64_b6_auto_2025 = auto_from_chi(fthmc_combined_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1029 = 1 / (1 - fthmc_combined_L64_b6_auto_1029[idx])
gamma_fthmc_1107 = 1 / (1 - fthmc_combined_L64_b6_auto_1107[idx])
gamma_fthmc_1331 = 1 / (1 - fthmc_combined_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_combined_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_combined_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_combined_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_combined_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_combined_L64_b6_auto_2025[idx])

gamma_ratio_1029 = gamma_hmc / gamma_fthmc_1029
gamma_ratio_1107 = gamma_hmc / gamma_fthmc_1107
gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1029, gamma_ratio_1107, gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

combined_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)

print("\n>>> combined b6 L64 gamma ratio")
print(f"mean({idx}) for combined b6 L64: {gv.mean(combined_L64_b6_gamma_ratio)}")
print(f"std({idx}) for combined b6 L64: {gv.sdev(combined_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]
deltaQ_fthmc_1029 = [ abs(fthmc_combined_L64_b6_topo_1029[i] - fthmc_combined_L64_b6_topo_1029[i-1]) for i in range(1, len(fthmc_combined_L64_b6_topo_1029))]
deltaQ_fthmc_1107 = [ abs(fthmc_combined_L64_b6_topo_1107[i] - fthmc_combined_L64_b6_topo_1107[i-1]) for i in range(1, len(fthmc_combined_L64_b6_topo_1107))]
deltaQ_fthmc_1331 = [ abs(fthmc_combined_L64_b6_topo_1331[i] - fthmc_combined_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_combined_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_combined_L64_b6_topo_1984[i] - fthmc_combined_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_combined_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_combined_L64_b6_topo_1999[i] - fthmc_combined_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_combined_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_combined_L64_b6_topo_2008[i] - fthmc_combined_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_combined_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_combined_L64_b6_topo_2017[i] - fthmc_combined_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_combined_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_combined_L64_b6_topo_2025[i] - fthmc_combined_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_combined_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1029), np.mean(deltaQ_fthmc_1107), np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

combined_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> combined b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for combined b6 L64: {gv.mean(combined_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for combined b6 L64: {gv.sdev(combined_L64_b6_deltaQ_ratio)}")




# %%
#! summary


gamma_ratio_ls = [base_L32_b6_gamma_ratio, attn_L32_b6_gamma_ratio, resn_L32_b6_gamma_ratio, tanh_L32_b6_gamma_ratio, base32_L32_b6_gamma_ratio, combined_L32_b6_gamma_ratio, gv.gvar(3.1119499568593634, 0.4234993124792864)]

deltaQ_ratio_ls = [base_L32_b6_deltaQ_ratio, attn_L32_b6_deltaQ_ratio, resn_L32_b6_deltaQ_ratio, tanh_L32_b6_deltaQ_ratio, base32_L32_b6_deltaQ_ratio, combined_L32_b6_deltaQ_ratio, gv.gvar(2.5406976744186047, 0.1752885282753643)]

fig, (ax1, ax2) = default_sub_plot()
# Adjust subplot spacing
plt.subplots_adjust(left=0.1,    # Increase left margin
                    right=0.95,    # Decrease right margin
                    bottom=0.15,   # Increase bottom margin 
                    top=0.95)      # Decrease top margin

ax1.errorbar(np.arange(len(gamma_ratio_ls)), [gv.mean(gamma_ratio) for gamma_ratio in gamma_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_ratio_ls], label="$\\beta=6.0$, $L=32$", **errorb)
ax2.errorbar(np.arange(len(deltaQ_ratio_ls)), [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], **errorb_circle)
ax1.set_ylabel('$R_{\\gamma (\\delta =16)}$', **fs_p)
ax1.set_ylim(1, 4.5)
# ax2.set_xlabel('Model', **fs_p)
ax2.set_ylabel('$R_{\\Delta Q}$', **fs_p)
ax2.set_ylim(1.4, 3.2)
ax2.set_xticks(np.arange(len(gamma_ratio_ls)))
ax2.set_xticklabels(['base', 'attn', 'resn', 'tanh', 'base32', 'combined', 'stable32'], fontsize=20)
ax1.legend(ncol=2, loc='upper right', **fs_small_p)
# plt.tight_layout()
plt.savefig('plots/summary_L32_b6_train_b3_L32.pdf', transparent=True)
plt.show()


# %%
gamma_ratio_ls = [base_L64_b6_gamma_ratio, attn_L64_b6_gamma_ratio, resn_L64_b6_gamma_ratio, tanh_L64_b6_gamma_ratio, base32_L64_b6_gamma_ratio, combined_L64_b6_gamma_ratio]

deltaQ_ratio_ls = [base_L64_b6_deltaQ_ratio, attn_L64_b6_deltaQ_ratio, resn_L64_b6_deltaQ_ratio, tanh_L64_b6_deltaQ_ratio, base32_L64_b6_deltaQ_ratio, combined_L64_b6_deltaQ_ratio]

fig, (ax1, ax2) = default_sub_plot()
# Adjust subplot spacing
plt.subplots_adjust(left=0.1,    # Increase left margin
                    right=0.95,    # Decrease right margin
                    bottom=0.15,   # Increase bottom margin 
                    top=0.95)      # Decrease top margin

ax1.errorbar(np.arange(len(gamma_ratio_ls)), [gv.mean(gamma_ratio) for gamma_ratio in gamma_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_ratio_ls], label="$\\beta=6.0$, $L=64$", **errorb)
ax2.errorbar(np.arange(len(deltaQ_ratio_ls)), [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], **errorb_circle)
ax1.set_ylabel('$R_{\\gamma (\\delta =16)}$', **fs_p)
ax1.set_ylim(1, 4.5)
# ax2.set_xlabel('Model', **fs_p)
ax2.set_ylabel('$R_{\\Delta Q}$', **fs_p)
ax2.set_ylim(1.5, 3.8)
ax2.set_xticks(np.arange(len(gamma_ratio_ls)))
ax2.set_xticklabels(['base', 'attn', 'resn', 'tanh', 'base32', 'combined'], fontsize=20)
ax1.legend(ncol=2, loc='upper right', **fs_small_p)
# plt.tight_layout()
plt.savefig('plots/summary_L64_b6_train_b3_L32.pdf', transparent=True)
plt.show()
# %%
