import numpy as np
import matplotlib.pyplot as plt
from HLTClass.SingleTauDataset import SingleTauDataset
from statsmodels.stats.proportion import proportion_confint

def compute_eff_witherr(N_num, N_den):
    conf_int = proportion_confint(count=N_num, nobs=N_den, alpha=0.32, method='beta')
    eff = N_num / N_den
    err_low = eff - conf_int[0]
    err_up = conf_int[1] - eff
    return eff, err_low, err_up

path_to_file = '/afs/cern.ch/work/p/pdebryas/PNetAtHLT/data/RateDen_nPV/Run_362617/EphemeralHLTPhysics0_Run2022G.root'
nPV_range = np.arange(8)
PNetparam = {
    'VTight':[1.1, 1.1, 130],
    'Tight':[0.98, 0.94, 130],
    'Medium':[0.94, 0.90, 130],
    'Loose':[0.88, 0.83, 130]
}
L1A_physics = 91374.04

# set up var
eph_dataset = SingleTauDataset(path_to_file)
save_rate = {}
print('ok')

#Compute for DeepTau
save_rate['DeepTau'] = {}

rate = []
rate_err = []
for nPV in nPV_range:
    N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3_nPV(nPV)
    if N_den_i != 0:
        rate_i, rate_low_i, rate_up_i  = compute_eff_witherr(N_num_i, N_den_i)
    else:
        rate_i, rate_low_i, rate_up_i  = 0, 0, 0
    rate.append(rate_i*L1A_physics)
    rate_err.append([rate_low_i*L1A_physics,rate_up_i*L1A_physics])

save_rate['DeepTau']['rate'] = rate
save_rate['DeepTau']['rate_err'] = rate_err

#Compute for PNet
for WP in PNetparam.keys():
    save_rate[WP] = {}
    rate = []
    rate_err = []
    for nPV in nPV_range:
        N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_SingleTauPNet_nPV(PNetparam[WP], nPV)
        if N_den_i != 0:
            rate_i, rate_low_i, rate_up_i  = compute_eff_witherr(N_num_i, N_den_i)
        else:
            rate_i, rate_low_i, rate_up_i  = 0, 0, 0
        rate.append(rate_i*L1A_physics)
        rate_err.append([rate_low_i*L1A_physics,rate_up_i*L1A_physics])

    save_rate[WP]['rate'] = rate
    save_rate[WP]['rate_err'] = rate_err


print(save_rate)

yerr_DeepTau = np.array(save_rate['DeepTau']['rate_err']).T
plt.errorbar(nPV_range, save_rate['DeepTau']['rate'], yerr=yerr_DeepTau, marker='+', label='DeepTau')
for WP in PNetparam.keys():
    yerr_DeepTau = np.array(save_rate[WP]['rate_err']).T
    plt.errorbar(nPV_range, save_rate[WP]['rate'], yerr=yerr_DeepTau, marker='+', label=WP)

plt.xlabel('nPV', fontsize = 17)
plt.ylabel('rate [Hz]', fontsize = 17)
plt.grid("on")
plt.legend(prop={'size': 17})
plt.savefig('figures/rate_nPV_SingleTau.pdf')
plt.close()
