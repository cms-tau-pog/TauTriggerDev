from helpers import compute_eff_witherr, load_cfg_file
import os 
import uproot
import awkward as ak

config = load_cfg_file()
HLT_name = config['HLT']['HLTname']
PNet_treshold = config['OPT']['PNet_treshold']
MCDataFolderNames = list(filter(None, (x.strip() for x in config['DATA']['MCDataFolderNames'].splitlines())))
if PNet_treshold == 'None':
    PNetConfig = False
    output_path = os.path.join(config['DATA']['result_eff'], HLT_name)
else:
    PNetConfig = True
    PNetTreshold = float(PNet_treshold)     
    output_path = os.path.join(config['DATA']['result_eff'], f'PNetTresh_{PNetTreshold}')

N_den = 0
N_num = 0
for i in range(len(MCDataFolderNames)):
    fileName = os.path.join(output_path, f'{MCDataFolderNames[i]}.root') 
    file = uproot.open(fileName)

    TausDen_pt = ak.flatten(file['TausDen']['Tau_pt'].array())
    TausNum_pt = ak.flatten(file['TausNum']['Tau_pt'].array())

    N_den += len(TausDen_pt)
    N_num += len(TausNum_pt)


eff, err_low, err_up = compute_eff_witherr(N_num, N_den)

print(f"Total number of hadronic GenTaus with with vis. pt <= 20 and eta< 2.1: {N_den}")
if PNetConfig:
    print(f"Total number of GenTaus matching with Jet passing {HLT_name} requirements with Pnet: {N_num}")
else:
    print(f"Total number of GenTaus matching with Tau passing {HLT_name} requirements: {N_num}")

print('Computed rate: ')
print(f"Eff : {eff}")
print(f"Eff_up : {err_low}")
print(f"Eff_down : {err_up}")