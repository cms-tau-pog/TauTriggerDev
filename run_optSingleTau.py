from helpers import files_from_path, load_cfg_file
import os
from scipy.optimize import minimize
import math
from HLTClass.SingleTauDataset import SingleTauDataset
import numpy as np

def loss(rate, rate_budget = 17):
    k = math.log(2) / 0.1
    if rate <= rate_budget:
        return 0
    if rate > (rate_budget + 0.1):
        return 1
    return math.exp(k * (rate - rate_budget)) - 1

def run_optimization(dataset_rate, dataset_eff):
    def f(a):
        # Pnet_chargeprob > 0
        a = np.append(a, [0])
        N_den, N_num = dataset_rate.get_Nnum_Nden_SingleTauPNet(a)
        rate = (N_num/N_den)*L1A_physics

        N_den, N_num = dataset_eff.ComputeEffAlgo_SingleTauPNet(a)
        eff_algo = (N_num/N_den)
        print(f'Rate: {rate}')
        print(f'Algo Eff: {eff_algo}') 
        print(f'Output score: {- eff_algo + loss(rate)}')
        print('---------------------------------------------------------------------------------------------------------')
        return - eff_algo + loss(rate)

    res = minimize(f, [0.70, 0.70], bounds=((0.5, 0.9), (0.5, 0.9)), method="L-BFGS-B", options={"eps": [0.005, 0.005]})

    return res.x

if __name__ == '__main__':

    config = load_cfg_file()
    RefRun = int(config['RUNINFO']['ref_run'])
    HLT_name = config['HLT']['HLTname']
    L1A_physics = float(config['RUNINFO']['L1A_physics'])

    Rate_path = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    Eff_path = os.path.join(config['DATA']['EffDenPath'], HLT_name)

    FileNameList_eff = "/afs/cern.ch/work/p/pdebryas/PNetAtHLT/data/EfficiencyDen/HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3/ZprimeToTauTau_M-4000.root"
    FileNameList_rate = files_from_path(Rate_path)[0] # only one otherwise it's too long (gives good aprosimation for the rate)

    dataset_eff = SingleTauDataset(FileNameList_eff)
    dataset_rate = SingleTauDataset(FileNameList_rate)

    optim_x = run_optimization(dataset_rate, dataset_eff)

    print('Optimisation finished:')
    print("Optimized parameters:", optim_x)
    optim_x = np.append(optim_x, [0])

    N_den, N_num = dataset_rate.get_Nnum_Nden_DiTauPNet(optim_x)
    rate_opt = (N_num/N_den)*L1A_physics

    print("Approx. Rate:", rate_opt)
