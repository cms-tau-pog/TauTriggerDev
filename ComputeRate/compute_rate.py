import json
from helpers import compute_eff_witherr, load_cfg_file
import os

# load config file info
config = load_cfg_file()
number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
L1A_physics = float(config['RUNINFO']['L1A_physics'])
RefRun = int(config['RUNINFO']['ref_run'])
LumiSectionsRange = [int(config['RUNINFO']['LumiSectionsRange_low']), int(config['RUNINFO']['LumiSectionsRange_up'])]
HLT_name = config['HLT']['HLTname']
HLT_rate = config['HLT']['HLT_rate']
runArea = config['RUNINFO']['Area']

PNet_treshold = config['OPT']['PNet_treshold']
if PNet_treshold == 'None':
    PNetConfig = False
    path_result = os.path.join(config['DATA']['result_rate'], f'result_{RefRun}', HLT_name)
else:
    PNetConfig = True
    PNetTreshold = float(PNet_treshold)
    path_result = os.path.join(config['DATA']['result_rate'], f'result_{RefRun}', f'PNetTresh_{PNetTreshold}/')

# compute sum of all Nevents in all the files
N_den = 0
N_num = 0
for i in range(number_of_ephemeral_folder):
    f = open(os.path.join(path_result, f'EphemeralHLTPhysics{i}_Run{runArea}.json'))
    data = json.load(f)
    for file in data.keys():
        N_den += data[file]['N_den']
        N_num += data[file]['N_num']
    f.close()

# compute rate here
eff, err_low, err_up = compute_eff_witherr(N_num, N_den)

print(f"Total number of events belonging to run {RefRun} and in LumiSections range {LumiSectionsRange}: {N_den}")

if PNetConfig:
    print(f"... and passing {HLT_name} conditions without DeepTau, and using PNet with treshold {PNetTreshold}: {N_num}")
else:
    print(f"... and passing {HLT_name} conditions: {N_num}")
    print(f"For comparison, {HLT_name} rate in cms oms is {HLT_rate}")

print('Computed rate: ')
print(f"Eff : {eff*L1A_physics}")
print(f"Eff_up : {err_low*L1A_physics}")
print(f"Eff_down : {err_up*L1A_physics}")