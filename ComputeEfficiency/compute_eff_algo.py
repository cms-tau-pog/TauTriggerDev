from helpers import load_cfg_file, compute_eff_witherr
import os

#compute Algo eff quickly

config = load_cfg_file()
HLT_name = config['HLT']['HLTname']
Eff_path = os.path.join(config['DATA']['EffDenPath'], HLT_name)

PNet_mode = config['OPT']['PNet_mode']
if PNet_mode == 'false':
    PNetMode = False
else:
    PNetMode = True
    PNetparam = [float(config['OPT']['PNet_t1']), float(config['OPT']['PNet_t2']), float(config['OPT']['PNet_t3'])]

# only one file otherwise it's too long
FileNameList_eff = "/afs/cern.ch/work/p/pdebryas/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1/VBFHToTauTau_M125_ext1.root" 

if HLT_name == 'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1':
    from HLTClass.DiTauDataset import DiTauDataset
    dataset_eff = DiTauDataset(FileNameList_eff)

    if PNetMode:
        print(f'Quick EffAlgo computation for DiTau path with PNet param {PNetparam}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_DiTauPNet(PNetparam)
    else:
        print(f'Quick EffAlgo computation for {HLT_name}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()       

EffAlgo, EffAlgo_low, EffAlgo_up = compute_eff_witherr(N_num, N_den)
print(f"Eff : {EffAlgo}")
print(f"Eff_up : {EffAlgo_up}")
print(f"Eff_down : {EffAlgo_low}")