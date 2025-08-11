# %%
import numpy as np
import sys
import gvar as gv
sys.path.append('/eagle/fthmc/run')
from Scaling_FT_HMC.utils.func import auto_from_chi
from lametlat.utils.plot_settings import *


# %%
#! base b3 L32

hmc_L32_b3_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta3.0.csv')

fthmc_base_L32_b3_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta3.0_base_train_b3.0_L32_1331.csv')
fthmc_base_L32_b3_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta3.0_base_train_b3.0_L32_1984.csv')
fthmc_base_L32_b3_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta3.0_base_train_b3.0_L32_1999.csv')
fthmc_base_L32_b3_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta3.0_base_train_b3.0_L32_2008.csv')
fthmc_base_L32_b3_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta3.0_base_train_b3.0_L32_2017.csv')
fthmc_base_L32_b3_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta3.0_base_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 3.0
volume = 32**2

hmc_L32_b3_auto = auto_from_chi(hmc_L32_b3_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_base_L32_b3_auto_1331 = auto_from_chi(fthmc_base_L32_b3_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b3_auto_1984 = auto_from_chi(fthmc_base_L32_b3_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b3_auto_1999 = auto_from_chi(fthmc_base_L32_b3_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b3_auto_2008 = auto_from_chi(fthmc_base_L32_b3_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b3_auto_2017 = auto_from_chi(fthmc_base_L32_b3_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b3_auto_2025 = auto_from_chi(fthmc_base_L32_b3_topo_2025, max_lag=max_lag, beta=beta, volume=volume)



gamma_hmc = 1 / (1 - hmc_L32_b3_auto[16])

gamma_fthmc_1331 = 1 / (1 - fthmc_base_L32_b3_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_base_L32_b3_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_base_L32_b3_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_base_L32_b3_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_base_L32_b3_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_base_L32_b3_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b3_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b3_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

base_L32_b3_gamma_ratio = gv.gvar(gamma_ratio_L32_b3_mean, gamma_ratio_L32_b3_std)

print(">>> base b3 L32 gamma ratio (16)")
print(f"mean(16) for base b3 L32: {gamma_ratio_L32_b3_mean}")
print(f"std(16) for base b3 L32: {gamma_ratio_L32_b3_std}")

deltaQ_hmc = [ abs(hmc_L32_b3_topo[i] - hmc_L32_b3_topo[i-1]) for i in range(1, len(hmc_L32_b3_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_base_L32_b3_topo_1331[i] - fthmc_base_L32_b3_topo_1331[i-1]) for i in range(1, len(fthmc_base_L32_b3_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_base_L32_b3_topo_1984[i] - fthmc_base_L32_b3_topo_1984[i-1]) for i in range(1, len(fthmc_base_L32_b3_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_base_L32_b3_topo_1999[i] - fthmc_base_L32_b3_topo_1999[i-1]) for i in range(1, len(fthmc_base_L32_b3_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_base_L32_b3_topo_2008[i] - fthmc_base_L32_b3_topo_2008[i-1]) for i in range(1, len(fthmc_base_L32_b3_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_base_L32_b3_topo_2017[i] - fthmc_base_L32_b3_topo_2017[i-1]) for i in range(1, len(fthmc_base_L32_b3_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_base_L32_b3_topo_2025[i] - fthmc_base_L32_b3_topo_2025[i-1]) for i in range(1, len(fthmc_base_L32_b3_topo_2025))]

deltaQ_hmc_L32_b3_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b3_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b3_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

base_L32_b3_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b3_mean / deltaQ_hmc_L32_b3_mean, deltaQ_fthmc_L32_b3_std / deltaQ_hmc_L32_b3_mean)

print("\n>>> base b3 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for base b3 L32: {gv.mean(base_L32_b3_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for base b3 L32: {gv.sdev(base_L32_b3_deltaQ_ratio)}")


# %%
#! base b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0.csv')

fthmc_base_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_base_train_b3.0_L32_1331.csv')
fthmc_base_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_base_train_b3.0_L32_1984.csv')
fthmc_base_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_base_train_b3.0_L32_1999.csv')
fthmc_base_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_base_train_b3.0_L32_2008.csv')
fthmc_base_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_base_train_b3.0_L32_2017.csv')
fthmc_base_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L32_beta6.0_base_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_base_L32_b6_auto_1331 = auto_from_chi(fthmc_base_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_1984 = auto_from_chi(fthmc_base_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_1999 = auto_from_chi(fthmc_base_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_2008 = auto_from_chi(fthmc_base_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_2017 = auto_from_chi(fthmc_base_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L32_b6_auto_2025 = auto_from_chi(fthmc_base_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1331 = 1 / (1 - fthmc_base_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_base_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_base_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_base_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_base_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_base_L32_b6_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

base_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> base b6 L32 gamma ratio")
print(f"mean(16) for base b6 L32: {gv.mean(base_L32_b6_gamma_ratio)}")
print(f"std(16) for base b6 L32: {gv.sdev(base_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_base_L32_b6_topo_1331[i] - fthmc_base_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_base_L32_b6_topo_1984[i] - fthmc_base_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_base_L32_b6_topo_1999[i] - fthmc_base_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_base_L32_b6_topo_2008[i] - fthmc_base_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_base_L32_b6_topo_2017[i] - fthmc_base_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_base_L32_b6_topo_2025[i] - fthmc_base_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_base_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

base_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> base b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for base b6 L32: {gv.mean(base_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for base b6 L32: {gv.sdev(base_L32_b6_deltaQ_ratio)}")

# %%
#! base b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0.csv')

fthmc_base_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_base_train_b3.0_L32_1331.csv')
fthmc_base_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_base_train_b3.0_L32_1984.csv')
fthmc_base_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_base_train_b3.0_L32_1999.csv')
fthmc_base_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_base_train_b3.0_L32_2008.csv')
fthmc_base_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_base_train_b3.0_L32_2017.csv')
fthmc_base_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_fthmc_L64_beta6.0_base_train_b3.0_L32_2025.csv')

max_lag = 200
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_base_L64_b6_auto_1331 = auto_from_chi(fthmc_base_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_1984 = auto_from_chi(fthmc_base_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_1999 = auto_from_chi(fthmc_base_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_2008 = auto_from_chi(fthmc_base_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_2017 = auto_from_chi(fthmc_base_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_base_L64_b6_auto_2025 = auto_from_chi(fthmc_base_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)



idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_base_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_base_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_base_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_base_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_base_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_base_L64_b6_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025]) 
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

base_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)


print("\n>>> base b6 L64 gamma ratio")
print(f"mean({idx}) for base b6 L64: {gv.mean(base_L64_b6_gamma_ratio)}")
print(f"std({idx}) for base b6 L64: {gv.sdev(base_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]

deltaQ_fthmc_1331 = [ abs(fthmc_base_L64_b6_topo_1331[i] - fthmc_base_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_base_L64_b6_topo_1984[i] - fthmc_base_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_base_L64_b6_topo_1999[i] - fthmc_base_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_base_L64_b6_topo_2008[i] - fthmc_base_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_base_L64_b6_topo_2017[i] - fthmc_base_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_base_L64_b6_topo_2025[i] - fthmc_base_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_base_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

base_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> base b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for base b6 L64: {gv.mean(base_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for base b6 L64: {gv.sdev(base_L64_b6_deltaQ_ratio)}")

# %%
#! attn b3 L32

hmc_L32_b3_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta3.0.csv')

fthmc_attn_L32_b3_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta3.0_attn_train_b3.0_L32_1331.csv')
fthmc_attn_L32_b3_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta3.0_attn_train_b3.0_L32_1984.csv')
fthmc_attn_L32_b3_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta3.0_attn_train_b3.0_L32_1999.csv')
fthmc_attn_L32_b3_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta3.0_attn_train_b3.0_L32_2008.csv')
fthmc_attn_L32_b3_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta3.0_attn_train_b3.0_L32_2017.csv')
fthmc_attn_L32_b3_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta3.0_attn_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 3.0
volume = 32**2

hmc_L32_b3_auto = auto_from_chi(hmc_L32_b3_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_attn_L32_b3_auto_1331 = auto_from_chi(fthmc_attn_L32_b3_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b3_auto_1984 = auto_from_chi(fthmc_attn_L32_b3_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b3_auto_1999 = auto_from_chi(fthmc_attn_L32_b3_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b3_auto_2008 = auto_from_chi(fthmc_attn_L32_b3_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b3_auto_2017 = auto_from_chi(fthmc_attn_L32_b3_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b3_auto_2025 = auto_from_chi(fthmc_attn_L32_b3_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b3_auto[16])

gamma_fthmc_1331 = 1 / (1 - fthmc_attn_L32_b3_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_attn_L32_b3_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_attn_L32_b3_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_attn_L32_b3_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_attn_L32_b3_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_attn_L32_b3_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b3_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b3_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])


attn_L32_b3_gamma_ratio = gv.gvar(gamma_ratio_L32_b3_mean, gamma_ratio_L32_b3_std)

print(">>> attn L32 b3 gamma ratio (16)")
print(f"mean(16) for attn L32 b3: {gv.mean(attn_L32_b3_gamma_ratio)}")
print(f"std(16) for attn L32 b3: {gv.sdev(attn_L32_b3_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b3_topo[i] - hmc_L32_b3_topo[i-1]) for i in range(1, len(hmc_L32_b3_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_attn_L32_b3_topo_1331[i] - fthmc_attn_L32_b3_topo_1331[i-1]) for i in range(1, len(fthmc_attn_L32_b3_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_attn_L32_b3_topo_1984[i] - fthmc_attn_L32_b3_topo_1984[i-1]) for i in range(1, len(fthmc_attn_L32_b3_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_attn_L32_b3_topo_1999[i] - fthmc_attn_L32_b3_topo_1999[i-1]) for i in range(1, len(fthmc_attn_L32_b3_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_attn_L32_b3_topo_2008[i] - fthmc_attn_L32_b3_topo_2008[i-1]) for i in range(1, len(fthmc_attn_L32_b3_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_attn_L32_b3_topo_2017[i] - fthmc_attn_L32_b3_topo_2017[i-1]) for i in range(1, len(fthmc_attn_L32_b3_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_attn_L32_b3_topo_2025[i] - fthmc_attn_L32_b3_topo_2025[i-1]) for i in range(1, len(fthmc_attn_L32_b3_topo_2025))]

deltaQ_hmc_L32_b3_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b3_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b3_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])


attn_L32_b3_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b3_mean / deltaQ_hmc_L32_b3_mean, deltaQ_fthmc_L32_b3_std / deltaQ_hmc_L32_b3_mean)

print("\n>>> attn L32 b3 deltaQ ratio")
print(f"mean(deltaQ) ratio for attn L32 b3: {gv.mean(attn_L32_b3_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for attn L32 b3: {gv.sdev(attn_L32_b3_deltaQ_ratio)}")


# %%
#! attn b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0.csv')

fthmc_attn_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_attn_train_b3.0_L32_1331.csv')
fthmc_attn_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_attn_train_b3.0_L32_1984.csv')
fthmc_attn_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_attn_train_b3.0_L32_1999.csv')
fthmc_attn_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_attn_train_b3.0_L32_2008.csv')
fthmc_attn_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_attn_train_b3.0_L32_2017.csv')
fthmc_attn_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L32_beta6.0_attn_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_attn_L32_b6_auto_1331 = auto_from_chi(fthmc_attn_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_1984 = auto_from_chi(fthmc_attn_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_1999 = auto_from_chi(fthmc_attn_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_2008 = auto_from_chi(fthmc_attn_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_2017 = auto_from_chi(fthmc_attn_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L32_b6_auto_2025 = auto_from_chi(fthmc_attn_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1331 = 1 / (1 - fthmc_attn_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_attn_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_attn_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_attn_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_attn_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_attn_L32_b6_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2025])

attn_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> attn b6 L32 gamma ratio")
print(f"mean(16) for attn b6 L32: {gv.mean(attn_L32_b6_gamma_ratio)}")
print(f"std(16) for attn b6 L32: {gv.sdev(attn_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_attn_L32_b6_topo_1331[i] - fthmc_attn_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_attn_L32_b6_topo_1984[i] - fthmc_attn_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_attn_L32_b6_topo_1999[i] - fthmc_attn_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_attn_L32_b6_topo_2008[i] - fthmc_attn_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_attn_L32_b6_topo_2017[i] - fthmc_attn_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_attn_L32_b6_topo_2025[i] - fthmc_attn_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_attn_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

attn_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> attn b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for attn b6 L32: {gv.mean(attn_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for attn b6 L32: {gv.sdev(attn_L32_b6_deltaQ_ratio)}")

# %%
#! attn b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0.csv')

fthmc_attn_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_attn_train_b3.0_L32_1331.csv')
fthmc_attn_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_attn_train_b3.0_L32_1984.csv')
fthmc_attn_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_attn_train_b3.0_L32_1999.csv')
fthmc_attn_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_attn_train_b3.0_L32_2008.csv')
fthmc_attn_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_attn_train_b3.0_L32_2017.csv')
fthmc_attn_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/attn_evaluation/dumps/topo_fthmc_L64_beta6.0_attn_train_b3.0_L32_2025.csv')

max_lag = 200
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_attn_L64_b6_auto_1331 = auto_from_chi(fthmc_attn_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_1984 = auto_from_chi(fthmc_attn_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_1999 = auto_from_chi(fthmc_attn_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_2008 = auto_from_chi(fthmc_attn_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_2017 = auto_from_chi(fthmc_attn_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_attn_L64_b6_auto_2025 = auto_from_chi(fthmc_attn_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_attn_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_attn_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_attn_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_attn_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_attn_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_attn_L64_b6_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

attn_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)


print("\n>>> attn b6 L64 gamma ratio")
print(f"mean({idx}) for attn b6 L64: {gv.mean(attn_L64_b6_gamma_ratio)}")
print(f"std({idx}) for attn b6 L64: {gv.sdev(attn_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]

deltaQ_fthmc_1331 = [ abs(fthmc_attn_L64_b6_topo_1331[i] - fthmc_attn_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_attn_L64_b6_topo_1984[i] - fthmc_attn_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_attn_L64_b6_topo_1999[i] - fthmc_attn_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_attn_L64_b6_topo_2008[i] - fthmc_attn_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_attn_L64_b6_topo_2017[i] - fthmc_attn_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_attn_L64_b6_topo_2025[i] - fthmc_attn_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_attn_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])


attn_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> attn b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for attn b6 L64: {gv.mean(attn_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for attn b6 L64: {gv.sdev(attn_L64_b6_deltaQ_ratio)}")

# %%
#! resn b3 L32

hmc_L32_b3_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta3.0.csv')

fthmc_resn_L32_b3_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta3.0_resn_train_b3.0_L32_1331.csv')
fthmc_resn_L32_b3_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta3.0_resn_train_b3.0_L32_1984.csv')
fthmc_resn_L32_b3_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta3.0_resn_train_b3.0_L32_1999.csv')
fthmc_resn_L32_b3_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta3.0_resn_train_b3.0_L32_2008.csv')
fthmc_resn_L32_b3_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta3.0_resn_train_b3.0_L32_2017.csv')
fthmc_resn_L32_b3_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta3.0_resn_train_b3.0_L32_2025.csv')

beta = 3.0
max_lag = 200
volume = 32**2

hmc_L32_b3_auto = auto_from_chi(hmc_L32_b3_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_resn_L32_b3_auto_1331 = auto_from_chi(fthmc_resn_L32_b3_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b3_auto_1984 = auto_from_chi(fthmc_resn_L32_b3_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b3_auto_1999 = auto_from_chi(fthmc_resn_L32_b3_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b3_auto_2008 = auto_from_chi(fthmc_resn_L32_b3_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b3_auto_2017 = auto_from_chi(fthmc_resn_L32_b3_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b3_auto_2025 = auto_from_chi(fthmc_resn_L32_b3_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b3_auto[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_resn_L32_b3_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_resn_L32_b3_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_resn_L32_b3_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_resn_L32_b3_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_resn_L32_b3_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_resn_L32_b3_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b3_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b3_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

resn_L32_b3_gamma_ratio = gv.gvar(gamma_ratio_L32_b3_mean, gamma_ratio_L32_b3_std)

print("\n>>> resn L32 b3 gamma ratio")
print(f"mean(16) for resn L32 b3: {gv.mean(resn_L32_b3_gamma_ratio)}")
print(f"std(16) for resn L32 b3: {gv.sdev(resn_L32_b3_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b3_topo[i] - hmc_L32_b3_topo[i-1]) for i in range(1, len(hmc_L32_b3_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_resn_L32_b3_topo_1331[i] - fthmc_resn_L32_b3_topo_1331[i-1]) for i in range(1, len(fthmc_resn_L32_b3_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_resn_L32_b3_topo_1984[i] - fthmc_resn_L32_b3_topo_1984[i-1]) for i in range(1, len(fthmc_resn_L32_b3_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_resn_L32_b3_topo_1999[i] - fthmc_resn_L32_b3_topo_1999[i-1]) for i in range(1, len(fthmc_resn_L32_b3_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_resn_L32_b3_topo_2008[i] - fthmc_resn_L32_b3_topo_2008[i-1]) for i in range(1, len(fthmc_resn_L32_b3_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_resn_L32_b3_topo_2017[i] - fthmc_resn_L32_b3_topo_2017[i-1]) for i in range(1, len(fthmc_resn_L32_b3_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_resn_L32_b3_topo_2025[i] - fthmc_resn_L32_b3_topo_2025[i-1]) for i in range(1, len(fthmc_resn_L32_b3_topo_2025))]

deltaQ_hmc_L32_b3_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b3_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b3_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

resn_L32_b3_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b3_mean / deltaQ_hmc_L32_b3_mean, deltaQ_fthmc_L32_b3_std / deltaQ_hmc_L32_b3_mean)

print("\n>>> resn L32 b3 deltaQ ratio")
print(f"mean(deltaQ) ratio for resn L32 b3: {gv.mean(resn_L32_b3_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for resn L32 b3: {gv.sdev(resn_L32_b3_deltaQ_ratio)}")

# %%
#! resn b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0.csv')

fthmc_resn_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_resn_train_b3.0_L32_1331.csv')
fthmc_resn_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_resn_train_b3.0_L32_1984.csv')
fthmc_resn_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_resn_train_b3.0_L32_1999.csv')
fthmc_resn_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_resn_train_b3.0_L32_2008.csv')
fthmc_resn_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_resn_train_b3.0_L32_2017.csv')
fthmc_resn_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L32_beta6.0_resn_train_b3.0_L32_2025.csv')

beta = 6.0
max_lag = 200
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_resn_L32_b6_auto_1331 = auto_from_chi(fthmc_resn_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_1984 = auto_from_chi(fthmc_resn_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_1999 = auto_from_chi(fthmc_resn_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_2008 = auto_from_chi(fthmc_resn_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_2017 = auto_from_chi(fthmc_resn_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L32_b6_auto_2025 = auto_from_chi(fthmc_resn_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)



gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])
gamma_fthmc_1331 = 1 / (1 - fthmc_resn_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_resn_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_resn_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_resn_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_resn_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_resn_L32_b6_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

resn_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> resn b6 L32 gamma ratio")
print(f"mean(16) for resn b6 L32: {gv.mean(resn_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_resn_L32_b6_topo_1331[i] - fthmc_resn_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_resn_L32_b6_topo_1984[i] - fthmc_resn_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_resn_L32_b6_topo_1999[i] - fthmc_resn_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_resn_L32_b6_topo_2008[i] - fthmc_resn_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_resn_L32_b6_topo_2017[i] - fthmc_resn_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_resn_L32_b6_topo_2025[i] - fthmc_resn_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_resn_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

resn_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> resn b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for resn b6 L32: {gv.mean(resn_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for resn b6 L32: {gv.sdev(resn_L32_b6_deltaQ_ratio)}")

# %%
#! resn b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0.csv')

fthmc_resn_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_resn_train_b3.0_L32_1331.csv')
fthmc_resn_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_resn_train_b3.0_L32_1984.csv')
fthmc_resn_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_resn_train_b3.0_L32_1999.csv')
fthmc_resn_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_resn_train_b3.0_L32_2008.csv')
fthmc_resn_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_resn_train_b3.0_L32_2017.csv')
fthmc_resn_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/resn_evaluation/dumps/topo_fthmc_L64_beta6.0_resn_train_b3.0_L32_2025.csv')

max_lag = 200
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_resn_L64_b6_auto_1331 = auto_from_chi(fthmc_resn_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_1984 = auto_from_chi(fthmc_resn_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_1999 = auto_from_chi(fthmc_resn_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_2008 = auto_from_chi(fthmc_resn_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_2017 = auto_from_chi(fthmc_resn_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_resn_L64_b6_auto_2025 = auto_from_chi(fthmc_resn_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 100

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_resn_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_resn_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_resn_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_resn_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_resn_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_resn_L64_b6_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

resn_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)

print("\n>>> resn b6 L64 gamma ratio")
print(f"mean({idx}) for resn b6 L64: {gv.mean(resn_L64_b6_gamma_ratio)}")
print(f"std({idx}) for resn b6 L64: {gv.sdev(resn_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_resn_L64_b6_topo_1331[i] - fthmc_resn_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_resn_L64_b6_topo_1984[i] - fthmc_resn_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_resn_L64_b6_topo_1999[i] - fthmc_resn_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_resn_L64_b6_topo_2008[i] - fthmc_resn_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_resn_L64_b6_topo_2017[i] - fthmc_resn_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_resn_L64_b6_topo_2025[i] - fthmc_resn_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_resn_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

resn_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> resn b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for resn b6 L64: {gv.mean(resn_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for resn b6 L64: {gv.sdev(resn_L64_b6_deltaQ_ratio)}")


# %%
#! coorconv b3 L32

hmc_L32_b3_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta3.0.csv')

fthmc_coorconv_L32_b3_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta3.0_coorconv_train_b3.0_L32_1331.csv')
fthmc_coorconv_L32_b3_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta3.0_coorconv_train_b3.0_L32_1984.csv')
fthmc_coorconv_L32_b3_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta3.0_coorconv_train_b3.0_L32_1999.csv')
fthmc_coorconv_L32_b3_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta3.0_coorconv_train_b3.0_L32_2008.csv')
fthmc_coorconv_L32_b3_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta3.0_coorconv_train_b3.0_L32_2017.csv')
fthmc_coorconv_L32_b3_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta3.0_coorconv_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 3.0
volume = 32**2

hmc_L32_b3_auto = auto_from_chi(hmc_L32_b3_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_coorconv_L32_b3_auto_1331 = auto_from_chi(fthmc_coorconv_L32_b3_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b3_auto_1984 = auto_from_chi(fthmc_coorconv_L32_b3_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b3_auto_1999 = auto_from_chi(fthmc_coorconv_L32_b3_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b3_auto_2008 = auto_from_chi(fthmc_coorconv_L32_b3_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b3_auto_2017 = auto_from_chi(fthmc_coorconv_L32_b3_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b3_auto_2025 = auto_from_chi(fthmc_coorconv_L32_b3_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b3_auto[16])

gamma_fthmc_1331 = 1 / (1 - fthmc_coorconv_L32_b3_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_coorconv_L32_b3_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_coorconv_L32_b3_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_coorconv_L32_b3_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_coorconv_L32_b3_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_coorconv_L32_b3_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b3_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b3_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

coorconv_L32_b3_gamma_ratio = gv.gvar(gamma_ratio_L32_b3_mean, gamma_ratio_L32_b3_std)

print("\n>>> coorconv L32 b3 gamma ratio (16)")
print(f"mean(16) for coorconv L32 b3: {gv.mean(coorconv_L32_b3_gamma_ratio)}")
print(f"std(16) for coorconv L32 b3: {gv.sdev(coorconv_L32_b3_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b3_topo[i] - hmc_L32_b3_topo[i-1]) for i in range(1, len(hmc_L32_b3_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_coorconv_L32_b3_topo_1331[i] - fthmc_coorconv_L32_b3_topo_1331[i-1]) for i in range(1, len(fthmc_coorconv_L32_b3_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_coorconv_L32_b3_topo_1984[i] - fthmc_coorconv_L32_b3_topo_1984[i-1]) for i in range(1, len(fthmc_coorconv_L32_b3_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_coorconv_L32_b3_topo_1999[i] - fthmc_coorconv_L32_b3_topo_1999[i-1]) for i in range(1, len(fthmc_coorconv_L32_b3_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_coorconv_L32_b3_topo_2008[i] - fthmc_coorconv_L32_b3_topo_2008[i-1]) for i in range(1, len(fthmc_coorconv_L32_b3_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_coorconv_L32_b3_topo_2017[i] - fthmc_coorconv_L32_b3_topo_2017[i-1]) for i in range(1, len(fthmc_coorconv_L32_b3_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_coorconv_L32_b3_topo_2025[i] - fthmc_coorconv_L32_b3_topo_2025[i-1]) for i in range(1, len(fthmc_coorconv_L32_b3_topo_2025))]

deltaQ_hmc_L32_b3_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b3_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b3_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

coorconv_L32_b3_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b3_mean / deltaQ_hmc_L32_b3_mean, deltaQ_fthmc_L32_b3_std / deltaQ_hmc_L32_b3_mean)

print("\n>>> coorconv L32 b3 deltaQ ratio")
print(f"mean(deltaQ) ratio for coorconv L32 b3: {gv.mean(coorconv_L32_b3_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for coorconv L32 b3: {gv.sdev(coorconv_L32_b3_deltaQ_ratio)}")

# %%
#! coorconv b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0.csv')

fthmc_coorconv_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta6.0_coorconv_train_b3.0_L32_1331.csv')
fthmc_coorconv_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta6.0_coorconv_train_b3.0_L32_1984.csv')
fthmc_coorconv_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta6.0_coorconv_train_b3.0_L32_1999.csv')
fthmc_coorconv_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta6.0_coorconv_train_b3.0_L32_2008.csv')
fthmc_coorconv_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta6.0_coorconv_train_b3.0_L32_2017.csv')
fthmc_coorconv_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L32_beta6.0_coorconv_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_coorconv_L32_b6_auto_1331 = auto_from_chi(fthmc_coorconv_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b6_auto_1984 = auto_from_chi(fthmc_coorconv_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b6_auto_1999 = auto_from_chi(fthmc_coorconv_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b6_auto_2008 = auto_from_chi(fthmc_coorconv_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b6_auto_2017 = auto_from_chi(fthmc_coorconv_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L32_b6_auto_2025 = auto_from_chi(fthmc_coorconv_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1331 = 1 / (1 - fthmc_coorconv_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_coorconv_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_coorconv_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_coorconv_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_coorconv_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_coorconv_L32_b6_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

coorconv_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> coorconv b6 L32 gamma ratio")
print(f"mean(16) for coorconv b6 L32: {gv.mean(coorconv_L32_b6_gamma_ratio)}")
print(f"std(16) for coorconv b6 L32: {gv.sdev(coorconv_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_coorconv_L32_b6_topo_1331[i] - fthmc_coorconv_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_coorconv_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_coorconv_L32_b6_topo_1984[i] - fthmc_coorconv_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_coorconv_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_coorconv_L32_b6_topo_1999[i] - fthmc_coorconv_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_coorconv_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_coorconv_L32_b6_topo_2008[i] - fthmc_coorconv_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_coorconv_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_coorconv_L32_b6_topo_2017[i] - fthmc_coorconv_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_coorconv_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_coorconv_L32_b6_topo_2025[i] - fthmc_coorconv_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_coorconv_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

coorconv_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> coorconv b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for coorconv b6 L32: {gv.mean(coorconv_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for coorconv b6 L32: {gv.sdev(coorconv_L32_b6_deltaQ_ratio)}")

# %%
#! coorconv b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0.csv')

fthmc_coorconv_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L64_beta6.0_coorconv_train_b3.0_L32_1331.csv')
fthmc_coorconv_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L64_beta6.0_coorconv_train_b3.0_L32_1984.csv')
fthmc_coorconv_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L64_beta6.0_coorconv_train_b3.0_L32_1999.csv')
fthmc_coorconv_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L64_beta6.0_coorconv_train_b3.0_L32_2008.csv')
fthmc_coorconv_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L64_beta6.0_coorconv_train_b3.0_L32_2017.csv')
fthmc_coorconv_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/coorconv_evaluation/dumps/topo_fthmc_L64_beta6.0_coorconv_train_b3.0_L32_2025.csv')

max_lag = 200
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_coorconv_L64_b6_auto_1331 = auto_from_chi(fthmc_coorconv_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L64_b6_auto_1984 = auto_from_chi(fthmc_coorconv_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L64_b6_auto_1999 = auto_from_chi(fthmc_coorconv_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L64_b6_auto_2008 = auto_from_chi(fthmc_coorconv_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L64_b6_auto_2017 = auto_from_chi(fthmc_coorconv_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_coorconv_L64_b6_auto_2025 = auto_from_chi(fthmc_coorconv_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_coorconv_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_coorconv_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_coorconv_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_coorconv_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_coorconv_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_coorconv_L64_b6_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

coorconv_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)

print("\n>>> coorconv b6 L64 gamma ratio")
print(f"mean({idx}) for coorconv b6 L64: {gv.mean(coorconv_L64_b6_gamma_ratio)}")
print(f"std({idx}) for coorconv b6 L64: {gv.sdev(coorconv_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]

deltaQ_fthmc_1331 = [ abs(fthmc_coorconv_L64_b6_topo_1331[i] - fthmc_coorconv_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_coorconv_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_coorconv_L64_b6_topo_1984[i] - fthmc_coorconv_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_coorconv_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_coorconv_L64_b6_topo_1999[i] - fthmc_coorconv_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_coorconv_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_coorconv_L64_b6_topo_2008[i] - fthmc_coorconv_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_coorconv_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_coorconv_L64_b6_topo_2017[i] - fthmc_coorconv_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_coorconv_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_coorconv_L64_b6_topo_2025[i] - fthmc_coorconv_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_coorconv_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

coorconv_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> coorconv b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for coorconv b6 L64: {gv.mean(coorconv_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for coorconv b6 L64: {gv.sdev(coorconv_L64_b6_deltaQ_ratio)}")

# %%
#! multif b3 L32

hmc_L32_b3_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta3.0.csv')

fthmc_multif_L32_b3_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta3.0_multif_train_b3.0_L32_1331.csv')
fthmc_multif_L32_b3_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta3.0_multif_train_b3.0_L32_1984.csv')
fthmc_multif_L32_b3_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta3.0_multif_train_b3.0_L32_1999.csv')
fthmc_multif_L32_b3_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta3.0_multif_train_b3.0_L32_2008.csv')
fthmc_multif_L32_b3_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta3.0_multif_train_b3.0_L32_2017.csv')
fthmc_multif_L32_b3_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta3.0_multif_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 3.0
volume = 32**2

hmc_L32_b3_auto = auto_from_chi(hmc_L32_b3_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_multif_L32_b3_auto_1331 = auto_from_chi(fthmc_multif_L32_b3_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b3_auto_1984 = auto_from_chi(fthmc_multif_L32_b3_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b3_auto_1999 = auto_from_chi(fthmc_multif_L32_b3_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b3_auto_2008 = auto_from_chi(fthmc_multif_L32_b3_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b3_auto_2017 = auto_from_chi(fthmc_multif_L32_b3_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b3_auto_2025 = auto_from_chi(fthmc_multif_L32_b3_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b3_auto[16])

gamma_fthmc_1331 = 1 / (1 - fthmc_multif_L32_b3_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_multif_L32_b3_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_multif_L32_b3_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_multif_L32_b3_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_multif_L32_b3_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_multif_L32_b3_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b3_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b3_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

multif_L32_b3_gamma_ratio = gv.gvar(gamma_ratio_L32_b3_mean, gamma_ratio_L32_b3_std)

print("\n>>> multif L32 b3 gamma ratio (16)")
print(f"mean(16) for multif L32 b3: {gv.mean(multif_L32_b3_gamma_ratio)}")
print(f"std(16) for multif L32 b3: {gv.sdev(multif_L32_b3_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b3_topo[i] - hmc_L32_b3_topo[i-1]) for i in range(1, len(hmc_L32_b3_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_multif_L32_b3_topo_1331[i] - fthmc_multif_L32_b3_topo_1331[i-1]) for i in range(1, len(fthmc_multif_L32_b3_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_multif_L32_b3_topo_1984[i] - fthmc_multif_L32_b3_topo_1984[i-1]) for i in range(1, len(fthmc_multif_L32_b3_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_multif_L32_b3_topo_1999[i] - fthmc_multif_L32_b3_topo_1999[i-1]) for i in range(1, len(fthmc_multif_L32_b3_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_multif_L32_b3_topo_2008[i] - fthmc_multif_L32_b3_topo_2008[i-1]) for i in range(1, len(fthmc_multif_L32_b3_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_multif_L32_b3_topo_2017[i] - fthmc_multif_L32_b3_topo_2017[i-1]) for i in range(1, len(fthmc_multif_L32_b3_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_multif_L32_b3_topo_2025[i] - fthmc_multif_L32_b3_topo_2025[i-1]) for i in range(1, len(fthmc_multif_L32_b3_topo_2025))]

deltaQ_hmc_L32_b3_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b3_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b3_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

multif_L32_b3_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b3_mean / deltaQ_hmc_L32_b3_mean, deltaQ_fthmc_L32_b3_std / deltaQ_hmc_L32_b3_mean)

print("\n>>> multif L32 b3 deltaQ ratio")
print(f"mean(deltaQ) ratio for multif L32 b3: {gv.mean(multif_L32_b3_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for multif L32 b3: {gv.sdev(multif_L32_b3_deltaQ_ratio)}")

# %%
#! multif b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0.csv')

fthmc_multif_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta6.0_multif_train_b3.0_L32_1331.csv')
fthmc_multif_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta6.0_multif_train_b3.0_L32_1984.csv')
fthmc_multif_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta6.0_multif_train_b3.0_L32_1999.csv')
fthmc_multif_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta6.0_multif_train_b3.0_L32_2008.csv')
fthmc_multif_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta6.0_multif_train_b3.0_L32_2017.csv')
fthmc_multif_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L32_beta6.0_multif_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_multif_L32_b6_auto_1331 = auto_from_chi(fthmc_multif_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b6_auto_1984 = auto_from_chi(fthmc_multif_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b6_auto_1999 = auto_from_chi(fthmc_multif_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b6_auto_2008 = auto_from_chi(fthmc_multif_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b6_auto_2017 = auto_from_chi(fthmc_multif_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L32_b6_auto_2025 = auto_from_chi(fthmc_multif_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)


gamma_hmc = 1 / (1 - hmc_L32_b6_auto[16])

gamma_fthmc_1331 = 1 / (1 - fthmc_multif_L32_b6_auto_1331[16])
gamma_fthmc_1984 = 1 / (1 - fthmc_multif_L32_b6_auto_1984[16])
gamma_fthmc_1999 = 1 / (1 - fthmc_multif_L32_b6_auto_1999[16])
gamma_fthmc_2008 = 1 / (1 - fthmc_multif_L32_b6_auto_2008[16])
gamma_fthmc_2017 = 1 / (1 - fthmc_multif_L32_b6_auto_2017[16])
gamma_fthmc_2025 = 1 / (1 - fthmc_multif_L32_b6_auto_2025[16])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

multif_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> multif b6 L32 gamma ratio")
print(f"mean(16) for multif b6 L32: {gv.mean(multif_L32_b6_gamma_ratio)}")
print(f"std(16) for multif b6 L32: {gv.sdev(multif_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_multif_L32_b6_topo_1331[i] - fthmc_multif_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_multif_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_multif_L32_b6_topo_1984[i] - fthmc_multif_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_multif_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_multif_L32_b6_topo_1999[i] - fthmc_multif_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_multif_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_multif_L32_b6_topo_2008[i] - fthmc_multif_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_multif_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_multif_L32_b6_topo_2017[i] - fthmc_multif_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_multif_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_multif_L32_b6_topo_2025[i] - fthmc_multif_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_multif_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

multif_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> multif b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for multif b6 L32: {gv.mean(multif_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for multif b6 L32: {gv.sdev(multif_L32_b6_deltaQ_ratio)}")

# %%
#! multif b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0.csv')

fthmc_multif_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L64_beta6.0_multif_train_b3.0_L32_1331.csv')
fthmc_multif_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L64_beta6.0_multif_train_b3.0_L32_1984.csv')
fthmc_multif_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L64_beta6.0_multif_train_b3.0_L32_1999.csv')
fthmc_multif_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L64_beta6.0_multif_train_b3.0_L32_2008.csv')
fthmc_multif_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L64_beta6.0_multif_train_b3.0_L32_2017.csv')
fthmc_multif_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/multif_evaluation/dumps/topo_fthmc_L64_beta6.0_multif_train_b3.0_L32_2025.csv')

max_lag = 200
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_multif_L64_b6_auto_1331 = auto_from_chi(fthmc_multif_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L64_b6_auto_1984 = auto_from_chi(fthmc_multif_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L64_b6_auto_1999 = auto_from_chi(fthmc_multif_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L64_b6_auto_2008 = auto_from_chi(fthmc_multif_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L64_b6_auto_2017 = auto_from_chi(fthmc_multif_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_multif_L64_b6_auto_2025 = auto_from_chi(fthmc_multif_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_multif_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_multif_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_multif_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_multif_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_multif_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_multif_L64_b6_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

multif_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)

print("\n>>> multif b6 L64 gamma ratio")
print(f"mean({idx}) for multif b6 L64: {gv.mean(multif_L64_b6_gamma_ratio)}")
print(f"std({idx}) for multif b6 L64: {gv.sdev(multif_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]

deltaQ_fthmc_1331 = [ abs(fthmc_multif_L64_b6_topo_1331[i] - fthmc_multif_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_multif_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_multif_L64_b6_topo_1984[i] - fthmc_multif_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_multif_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_multif_L64_b6_topo_1999[i] - fthmc_multif_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_multif_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_multif_L64_b6_topo_2008[i] - fthmc_multif_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_multif_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_multif_L64_b6_topo_2017[i] - fthmc_multif_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_multif_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_multif_L64_b6_topo_2025[i] - fthmc_multif_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_multif_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

multif_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> multif b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for multif b6 L64: {gv.mean(multif_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for multif b6 L64: {gv.sdev(multif_L64_b6_deltaQ_ratio)}")

# %%
#! alpha b3 L32

hmc_L32_b3_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta3.0.csv')

fthmc_alpha_L32_b3_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta3.0_alpha_train_b3.0_L32_1331.csv')
fthmc_alpha_L32_b3_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta3.0_alpha_train_b3.0_L32_1984.csv')
fthmc_alpha_L32_b3_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta3.0_alpha_train_b3.0_L32_1999.csv')
fthmc_alpha_L32_b3_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta3.0_alpha_train_b3.0_L32_2008.csv')
fthmc_alpha_L32_b3_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta3.0_alpha_train_b3.0_L32_2017.csv')
fthmc_alpha_L32_b3_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta3.0_alpha_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 3.0
volume = 32**2

hmc_L32_b3_auto = auto_from_chi(hmc_L32_b3_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_alpha_L32_b3_auto_1331 = auto_from_chi(fthmc_alpha_L32_b3_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b3_auto_1984 = auto_from_chi(fthmc_alpha_L32_b3_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b3_auto_1999 = auto_from_chi(fthmc_alpha_L32_b3_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b3_auto_2008 = auto_from_chi(fthmc_alpha_L32_b3_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b3_auto_2017 = auto_from_chi(fthmc_alpha_L32_b3_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b3_auto_2025 = auto_from_chi(fthmc_alpha_L32_b3_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L32_b3_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_alpha_L32_b3_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_alpha_L32_b3_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_alpha_L32_b3_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_alpha_L32_b3_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_alpha_L32_b3_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_alpha_L32_b3_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b3_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b3_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

alpha_L32_b3_gamma_ratio = gv.gvar(gamma_ratio_L32_b3_mean, gamma_ratio_L32_b3_std)

print("\n>>> alpha b3 L32 gamma ratio")
print(f"mean({idx}) for alpha b3 L32: {gv.mean(alpha_L32_b3_gamma_ratio)}")
print(f"std({idx}) for alpha b3 L32: {gv.sdev(alpha_L32_b3_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b3_topo[i] - hmc_L32_b3_topo[i-1]) for i in range(1, len(hmc_L32_b3_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_alpha_L32_b3_topo_1331[i] - fthmc_alpha_L32_b3_topo_1331[i-1]) for i in range(1, len(fthmc_alpha_L32_b3_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_alpha_L32_b3_topo_1984[i] - fthmc_alpha_L32_b3_topo_1984[i-1]) for i in range(1, len(fthmc_alpha_L32_b3_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_alpha_L32_b3_topo_1999[i] - fthmc_alpha_L32_b3_topo_1999[i-1]) for i in range(1, len(fthmc_alpha_L32_b3_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_alpha_L32_b3_topo_2008[i] - fthmc_alpha_L32_b3_topo_2008[i-1]) for i in range(1, len(fthmc_alpha_L32_b3_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_alpha_L32_b3_topo_2017[i] - fthmc_alpha_L32_b3_topo_2017[i-1]) for i in range(1, len(fthmc_alpha_L32_b3_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_alpha_L32_b3_topo_2025[i] - fthmc_alpha_L32_b3_topo_2025[i-1]) for i in range(1, len(fthmc_alpha_L32_b3_topo_2025))]

deltaQ_hmc_L32_b3_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b3_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b3_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

alpha_L32_b3_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b3_mean / deltaQ_hmc_L32_b3_mean, deltaQ_fthmc_L32_b3_std / deltaQ_hmc_L32_b3_mean)

print("\n>>> alpha b3 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for alpha b3 L32: {gv.mean(alpha_L32_b3_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for alpha b3 L32: {gv.sdev(alpha_L32_b3_deltaQ_ratio)}")

# %%
#! alpha b6 L32
hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0.csv')

fthmc_alpha_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta6.0_alpha_train_b3.0_L32_1331.csv')
fthmc_alpha_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta6.0_alpha_train_b3.0_L32_1984.csv')
fthmc_alpha_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta6.0_alpha_train_b3.0_L32_1999.csv')
fthmc_alpha_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta6.0_alpha_train_b3.0_L32_2008.csv')
fthmc_alpha_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta6.0_alpha_train_b3.0_L32_2017.csv')
fthmc_alpha_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L32_beta6.0_alpha_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_alpha_L32_b6_auto_1331 = auto_from_chi(fthmc_alpha_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b6_auto_1984 = auto_from_chi(fthmc_alpha_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b6_auto_1999 = auto_from_chi(fthmc_alpha_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b6_auto_2008 = auto_from_chi(fthmc_alpha_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b6_auto_2017 = auto_from_chi(fthmc_alpha_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L32_b6_auto_2025 = auto_from_chi(fthmc_alpha_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L32_b6_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_alpha_L32_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_alpha_L32_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_alpha_L32_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_alpha_L32_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_alpha_L32_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_alpha_L32_b6_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

alpha_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> alpha b6 L32 gamma ratio")
print(f"mean({idx}) for alpha b6 L32: {gv.mean(alpha_L32_b6_gamma_ratio)}")
print(f"std({idx}) for alpha b6 L32: {gv.sdev(alpha_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_alpha_L32_b6_topo_1331[i] - fthmc_alpha_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_alpha_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_alpha_L32_b6_topo_1984[i] - fthmc_alpha_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_alpha_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_alpha_L32_b6_topo_1999[i] - fthmc_alpha_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_alpha_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_alpha_L32_b6_topo_2008[i] - fthmc_alpha_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_alpha_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_alpha_L32_b6_topo_2017[i] - fthmc_alpha_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_alpha_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_alpha_L32_b6_topo_2025[i] - fthmc_alpha_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_alpha_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

alpha_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> alpha b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for alpha b6 L32: {gv.mean(alpha_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for alpha b6 L32: {gv.sdev(alpha_L32_b6_deltaQ_ratio)}")

# %%
#! alpha b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0.csv')

fthmc_alpha_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L64_beta6.0_alpha_train_b3.0_L32_1331.csv')
fthmc_alpha_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L64_beta6.0_alpha_train_b3.0_L32_1984.csv')
fthmc_alpha_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L64_beta6.0_alpha_train_b3.0_L32_1999.csv')
fthmc_alpha_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L64_beta6.0_alpha_train_b3.0_L32_2008.csv')
fthmc_alpha_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L64_beta6.0_alpha_train_b3.0_L32_2017.csv')
fthmc_alpha_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/alpha_evaluation/dumps/topo_fthmc_L64_beta6.0_alpha_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_alpha_L64_b6_auto_1331 = auto_from_chi(fthmc_alpha_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L64_b6_auto_1984 = auto_from_chi(fthmc_alpha_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L64_b6_auto_1999 = auto_from_chi(fthmc_alpha_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L64_b6_auto_2008 = auto_from_chi(fthmc_alpha_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L64_b6_auto_2017 = auto_from_chi(fthmc_alpha_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_alpha_L64_b6_auto_2025 = auto_from_chi(fthmc_alpha_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_alpha_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_alpha_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_alpha_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_alpha_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_alpha_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_alpha_L64_b6_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

alpha_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)

print("\n>>> alpha b6 L64 gamma ratio")
print(f"mean({idx}) for alpha b6 L64: {gv.mean(alpha_L64_b6_gamma_ratio)}")
print(f"std({idx}) for alpha b6 L64: {gv.sdev(alpha_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_alpha_L64_b6_topo_1331[i] - fthmc_alpha_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_alpha_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_alpha_L64_b6_topo_1984[i] - fthmc_alpha_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_alpha_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_alpha_L64_b6_topo_1999[i] - fthmc_alpha_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_alpha_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_alpha_L64_b6_topo_2008[i] - fthmc_alpha_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_alpha_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_alpha_L64_b6_topo_2017[i] - fthmc_alpha_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_alpha_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_alpha_L64_b6_topo_2025[i] - fthmc_alpha_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_alpha_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

alpha_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> alpha b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for alpha b6 L64: {gv.mean(alpha_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for alpha b6 L64: {gv.sdev(alpha_L64_b6_deltaQ_ratio)}")



# %%
#! arctan b3 L32

hmc_L32_b3_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta3.0.csv')

fthmc_arctan_L32_b3_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta3.0_arctan_train_b3.0_L32_1331.csv')
fthmc_arctan_L32_b3_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta3.0_arctan_train_b3.0_L32_1984.csv')
fthmc_arctan_L32_b3_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta3.0_arctan_train_b3.0_L32_1999.csv')
fthmc_arctan_L32_b3_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta3.0_arctan_train_b3.0_L32_2008.csv')
fthmc_arctan_L32_b3_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta3.0_arctan_train_b3.0_L32_2017.csv')
fthmc_arctan_L32_b3_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta3.0_arctan_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 3.0
volume = 32**2

hmc_L32_b3_auto = auto_from_chi(hmc_L32_b3_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_arctan_L32_b3_auto_1331 = auto_from_chi(fthmc_arctan_L32_b3_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b3_auto_1984 = auto_from_chi(fthmc_arctan_L32_b3_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b3_auto_1999 = auto_from_chi(fthmc_arctan_L32_b3_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b3_auto_2008 = auto_from_chi(fthmc_arctan_L32_b3_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b3_auto_2017 = auto_from_chi(fthmc_arctan_L32_b3_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b3_auto_2025 = auto_from_chi(fthmc_arctan_L32_b3_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L32_b3_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_arctan_L32_b3_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_arctan_L32_b3_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_arctan_L32_b3_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_arctan_L32_b3_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_arctan_L32_b3_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_arctan_L32_b3_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b3_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])
gamma_ratio_L32_b3_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2017, gamma_ratio_2025])

arctan_L32_b3_gamma_ratio = gv.gvar(gamma_ratio_L32_b3_mean, gamma_ratio_L32_b3_std)

print("\n>>> arctan b3 L32 gamma ratio")
print(f"mean({idx}) for arctan b3 L32: {gv.mean(arctan_L32_b3_gamma_ratio)}")
print(f"std({idx}) for arctan b3 L32: {gv.sdev(arctan_L32_b3_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b3_topo[i] - hmc_L32_b3_topo[i-1]) for i in range(1, len(hmc_L32_b3_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_arctan_L32_b3_topo_1331[i] - fthmc_arctan_L32_b3_topo_1331[i-1]) for i in range(1, len(fthmc_arctan_L32_b3_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_arctan_L32_b3_topo_1984[i] - fthmc_arctan_L32_b3_topo_1984[i-1]) for i in range(1, len(fthmc_arctan_L32_b3_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_arctan_L32_b3_topo_1999[i] - fthmc_arctan_L32_b3_topo_1999[i-1]) for i in range(1, len(fthmc_arctan_L32_b3_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_arctan_L32_b3_topo_2008[i] - fthmc_arctan_L32_b3_topo_2008[i-1]) for i in range(1, len(fthmc_arctan_L32_b3_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_arctan_L32_b3_topo_2017[i] - fthmc_arctan_L32_b3_topo_2017[i-1]) for i in range(1, len(fthmc_arctan_L32_b3_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_arctan_L32_b3_topo_2025[i] - fthmc_arctan_L32_b3_topo_2025[i-1]) for i in range(1, len(fthmc_arctan_L32_b3_topo_2025))]

deltaQ_hmc_L32_b3_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b3_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b3_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

arctan_L32_b3_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b3_mean / deltaQ_hmc_L32_b3_mean, deltaQ_fthmc_L32_b3_std / deltaQ_hmc_L32_b3_mean)

print("\n>>> arctan b3 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for arctan b3 L32: {gv.mean(arctan_L32_b3_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for arctan b3 L32: {gv.sdev(arctan_L32_b3_deltaQ_ratio)}")

# %%
#! arctan b6 L32

hmc_L32_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L32_beta6.0.csv')

fthmc_arctan_L32_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_arctan_train_b3.0_L32_1331.csv')
fthmc_arctan_L32_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_arctan_train_b3.0_L32_1984.csv')
fthmc_arctan_L32_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_arctan_train_b3.0_L32_1999.csv')
fthmc_arctan_L32_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_arctan_train_b3.0_L32_2008.csv')
fthmc_arctan_L32_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_arctan_train_b3.0_L32_2017.csv')
fthmc_arctan_L32_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L32_beta6.0_arctan_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 32**2

hmc_L32_b6_auto = auto_from_chi(hmc_L32_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_arctan_L32_b6_auto_1331 = auto_from_chi(fthmc_arctan_L32_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_1984 = auto_from_chi(fthmc_arctan_L32_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_1999 = auto_from_chi(fthmc_arctan_L32_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_2008 = auto_from_chi(fthmc_arctan_L32_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_2017 = auto_from_chi(fthmc_arctan_L32_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L32_b6_auto_2025 = auto_from_chi(fthmc_arctan_L32_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L32_b6_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_arctan_L32_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_arctan_L32_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_arctan_L32_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_arctan_L32_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_arctan_L32_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_arctan_L32_b6_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L32_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2025])
gamma_ratio_L32_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2025])

arctan_L32_b6_gamma_ratio = gv.gvar(gamma_ratio_L32_b6_mean, gamma_ratio_L32_b6_std)

print("\n>>> arctan b6 L32 gamma ratio")
print(f"mean({idx}) for arctan b6 L32: {gv.mean(arctan_L32_b6_gamma_ratio)}")
print(f"std({idx}) for arctan b6 L32: {gv.sdev(arctan_L32_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L32_b6_topo[i] - hmc_L32_b6_topo[i-1]) for i in range(1, len(hmc_L32_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_arctan_L32_b6_topo_1331[i] - fthmc_arctan_L32_b6_topo_1331[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_arctan_L32_b6_topo_1984[i] - fthmc_arctan_L32_b6_topo_1984[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_arctan_L32_b6_topo_1999[i] - fthmc_arctan_L32_b6_topo_1999[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_arctan_L32_b6_topo_2008[i] - fthmc_arctan_L32_b6_topo_2008[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_arctan_L32_b6_topo_2017[i] - fthmc_arctan_L32_b6_topo_2017[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_arctan_L32_b6_topo_2025[i] - fthmc_arctan_L32_b6_topo_2025[i-1]) for i in range(1, len(fthmc_arctan_L32_b6_topo_2025))]

deltaQ_hmc_L32_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L32_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L32_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

arctan_L32_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L32_b6_mean / deltaQ_hmc_L32_b6_mean, deltaQ_fthmc_L32_b6_std / deltaQ_hmc_L32_b6_mean)

print("\n>>> arctan b6 L32 deltaQ ratio")
print(f"mean(deltaQ) ratio for arctan b6 L32: {gv.mean(arctan_L32_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for arctan b6 L32: {gv.sdev(arctan_L32_b6_deltaQ_ratio)}")

# %%
#! arctan b6 L64

hmc_L64_b6_topo = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/scaling/dumps/topo_hmc_L64_beta6.0.csv')

fthmc_arctan_L64_b6_topo_1331 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_arctan_train_b3.0_L32_1331.csv')
fthmc_arctan_L64_b6_topo_1984 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_arctan_train_b3.0_L32_1984.csv')
fthmc_arctan_L64_b6_topo_1999 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_arctan_train_b3.0_L32_1999.csv')
fthmc_arctan_L64_b6_topo_2008 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_arctan_train_b3.0_L32_2008.csv')
fthmc_arctan_L64_b6_topo_2017 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_arctan_train_b3.0_L32_2017.csv')
fthmc_arctan_L64_b6_topo_2025 = np.loadtxt('/eagle/fthmc/run/Scaling_FT_HMC/arctan_evaluation/dumps/topo_fthmc_L64_beta6.0_arctan_train_b3.0_L32_2025.csv')

max_lag = 64
beta = 6.0
volume = 64**2

hmc_L64_b6_auto = auto_from_chi(hmc_L64_b6_topo, max_lag=max_lag, beta=beta, volume=volume)

fthmc_arctan_L64_b6_auto_1331 = auto_from_chi(fthmc_arctan_L64_b6_topo_1331, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_1984 = auto_from_chi(fthmc_arctan_L64_b6_topo_1984, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_1999 = auto_from_chi(fthmc_arctan_L64_b6_topo_1999, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_2008 = auto_from_chi(fthmc_arctan_L64_b6_topo_2008, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_2017 = auto_from_chi(fthmc_arctan_L64_b6_topo_2017, max_lag=max_lag, beta=beta, volume=volume)
fthmc_arctan_L64_b6_auto_2025 = auto_from_chi(fthmc_arctan_L64_b6_topo_2025, max_lag=max_lag, beta=beta, volume=volume)

idx = 16

gamma_hmc = 1 / (1 - hmc_L64_b6_auto[idx])

gamma_fthmc_1331 = 1 / (1 - fthmc_arctan_L64_b6_auto_1331[idx])
gamma_fthmc_1984 = 1 / (1 - fthmc_arctan_L64_b6_auto_1984[idx])
gamma_fthmc_1999 = 1 / (1 - fthmc_arctan_L64_b6_auto_1999[idx])
gamma_fthmc_2008 = 1 / (1 - fthmc_arctan_L64_b6_auto_2008[idx])
gamma_fthmc_2017 = 1 / (1 - fthmc_arctan_L64_b6_auto_2017[idx])
gamma_fthmc_2025 = 1 / (1 - fthmc_arctan_L64_b6_auto_2025[idx])

gamma_ratio_1331 = gamma_hmc / gamma_fthmc_1331
gamma_ratio_1984 = gamma_hmc / gamma_fthmc_1984
gamma_ratio_1999 = gamma_hmc / gamma_fthmc_1999
gamma_ratio_2008 = gamma_hmc / gamma_fthmc_2008
gamma_ratio_2017 = gamma_hmc / gamma_fthmc_2017
gamma_ratio_2025 = gamma_hmc / gamma_fthmc_2025

gamma_ratio_L64_b6_mean = np.mean([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2025])
gamma_ratio_L64_b6_std = np.std([gamma_ratio_1331, gamma_ratio_1984, gamma_ratio_1999, gamma_ratio_2008, gamma_ratio_2025])

arctan_L64_b6_gamma_ratio = gv.gvar(gamma_ratio_L64_b6_mean, gamma_ratio_L64_b6_std)

print("\n>>> arctan b6 L64 gamma ratio")
print(f"mean({idx}) for arctan b6 L64: {gv.mean(arctan_L64_b6_gamma_ratio)}")
print(f"std({idx}) for arctan b6 L64: {gv.sdev(arctan_L64_b6_gamma_ratio)}")

deltaQ_hmc = [ abs(hmc_L64_b6_topo[i] - hmc_L64_b6_topo[i-1]) for i in range(1, len(hmc_L64_b6_topo))]
deltaQ_fthmc_1331 = [ abs(fthmc_arctan_L64_b6_topo_1331[i] - fthmc_arctan_L64_b6_topo_1331[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_1331))]
deltaQ_fthmc_1984 = [ abs(fthmc_arctan_L64_b6_topo_1984[i] - fthmc_arctan_L64_b6_topo_1984[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_1984))]
deltaQ_fthmc_1999 = [ abs(fthmc_arctan_L64_b6_topo_1999[i] - fthmc_arctan_L64_b6_topo_1999[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_1999))]
deltaQ_fthmc_2008 = [ abs(fthmc_arctan_L64_b6_topo_2008[i] - fthmc_arctan_L64_b6_topo_2008[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_2008))]
deltaQ_fthmc_2017 = [ abs(fthmc_arctan_L64_b6_topo_2017[i] - fthmc_arctan_L64_b6_topo_2017[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_2017))]
deltaQ_fthmc_2025 = [ abs(fthmc_arctan_L64_b6_topo_2025[i] - fthmc_arctan_L64_b6_topo_2025[i-1]) for i in range(1, len(fthmc_arctan_L64_b6_topo_2025))]

deltaQ_hmc_L64_b6_mean = np.mean(deltaQ_hmc)
deltaQ_fthmc_L64_b6_mean = np.mean([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])
deltaQ_fthmc_L64_b6_std = np.std([np.mean(deltaQ_fthmc_1331), np.mean(deltaQ_fthmc_1984), np.mean(deltaQ_fthmc_1999), np.mean(deltaQ_fthmc_2008), np.mean(deltaQ_fthmc_2017), np.mean(deltaQ_fthmc_2025)])

arctan_L64_b6_deltaQ_ratio = gv.gvar(deltaQ_fthmc_L64_b6_mean / deltaQ_hmc_L64_b6_mean, deltaQ_fthmc_L64_b6_std / deltaQ_hmc_L64_b6_mean)

print("\n>>> arctan b6 L64 deltaQ ratio")
print(f"mean(deltaQ) ratio for arctan b6 L64: {gv.mean(arctan_L64_b6_deltaQ_ratio)}")
print(f"std(deltaQ) ratio for arctan b6 L64: {gv.sdev(arctan_L64_b6_deltaQ_ratio)}")





# %%
#! summary

gamma_ratio_ls = [base_L32_b3_gamma_ratio, attn_L32_b3_gamma_ratio, resn_L32_b3_gamma_ratio, coorconv_L32_b3_gamma_ratio, multif_L32_b3_gamma_ratio, alpha_L32_b3_gamma_ratio, arctan_L32_b3_gamma_ratio]

deltaQ_ratio_ls = [base_L32_b3_deltaQ_ratio, attn_L32_b3_deltaQ_ratio, resn_L32_b3_deltaQ_ratio, coorconv_L32_b3_deltaQ_ratio, multif_L32_b3_deltaQ_ratio, alpha_L32_b3_deltaQ_ratio, arctan_L32_b3_deltaQ_ratio]

fig, ax = default_plot()
ax.errorbar(np.arange(len(gamma_ratio_ls)), [gv.mean(gamma_ratio) for gamma_ratio in gamma_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_ratio_ls], label='gamma ratio', **errorb)
ax.errorbar(np.arange(len(deltaQ_ratio_ls)), [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], label='deltaQ ratio', **errorb)

ax.legend(ncol=2, loc='upper right', fontsize=12)
ax.set_xlabel('Model', **fs_p)
ax.set_ylabel('Ratio', **fs_p)
plt.title("Comparison L32 b3", **fs_p)
plt.tight_layout()
plt.show()


gamma_ratio_ls = [base_L32_b6_gamma_ratio, attn_L32_b6_gamma_ratio, resn_L32_b6_gamma_ratio, coorconv_L32_b6_gamma_ratio, multif_L32_b6_gamma_ratio, alpha_L32_b6_gamma_ratio, arctan_L32_b6_gamma_ratio]

deltaQ_ratio_ls = [base_L32_b6_deltaQ_ratio, attn_L32_b6_deltaQ_ratio, resn_L32_b6_deltaQ_ratio, coorconv_L32_b6_deltaQ_ratio, multif_L32_b6_deltaQ_ratio, alpha_L32_b6_deltaQ_ratio, arctan_L32_b6_deltaQ_ratio]

fig, ax = default_plot()
ax.errorbar(np.arange(len(gamma_ratio_ls)), [gv.mean(gamma_ratio) for gamma_ratio in gamma_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_ratio_ls], label='gamma ratio', **errorb)
ax.errorbar(np.arange(len(deltaQ_ratio_ls)), [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], label='deltaQ ratio', **errorb)

ax.legend(ncol=2, loc='upper right', fontsize=12)
ax.set_xlabel('Model', **fs_p)
ax.set_ylabel('Ratio', **fs_p)
plt.title("Comparison L32 b6", **fs_p)
plt.tight_layout()
plt.show()


gamma_ratio_ls = [base_L64_b6_gamma_ratio, attn_L64_b6_gamma_ratio, resn_L64_b6_gamma_ratio, coorconv_L64_b6_gamma_ratio, multif_L64_b6_gamma_ratio, alpha_L64_b6_gamma_ratio, arctan_L64_b6_gamma_ratio]

deltaQ_ratio_ls = [base_L64_b6_deltaQ_ratio, attn_L64_b6_deltaQ_ratio, resn_L64_b6_deltaQ_ratio, coorconv_L64_b6_deltaQ_ratio, multif_L64_b6_deltaQ_ratio, alpha_L64_b6_deltaQ_ratio, arctan_L64_b6_deltaQ_ratio]

fig, ax = default_plot()
ax.errorbar(np.arange(len(gamma_ratio_ls)), [gv.mean(gamma_ratio) for gamma_ratio in gamma_ratio_ls], [gv.sdev(gamma_ratio) for gamma_ratio in gamma_ratio_ls], label='gamma ratio', **errorb)
ax.errorbar(np.arange(len(deltaQ_ratio_ls)), [gv.mean(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], [gv.sdev(deltaQ_ratio) for deltaQ_ratio in deltaQ_ratio_ls], label='deltaQ ratio', **errorb)

ax.legend(ncol=2, loc='upper right', fontsize=12)
ax.set_xlabel('Model', **fs_p)
ax.set_ylabel('Ratio', **fs_p)
plt.title("Comparison L64 b6", **fs_p)
plt.tight_layout()
plt.show()
# %%
