#Comparison between measured Run 3 rates 
# - computed with Ephemeral data 
# - from nominal values (CMS OMS) for different HLT (diTau/singleTau)

# In result/ folder we have json file for all ephemeral data within folder_i (from 0 to [number_of_ephemeral_folder -1]) with:
#  - N_den: number of event that are in ref_run and within LumiSectionsRange
#  - N_num (which pass HLT) for all ephemeral data

import json
from helpers import compute_eff_witherr, load_cfg_file_ComputeRate
from tqdm import tqdm

config = load_cfg_file_ComputeRate()
number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
path_json = config['DATA']['output_path']
L1A_physics = float(config['RUNINFO']['L1A_physics'])
RefRun = int(config['RUNINFO']['ref_run'])
LumiSectionsRange_low = int(config['RUNINFO']['LumiSectionsRange_low'])
LumiSectionsRange_up = int(config['RUNINFO']['LumiSectionsRange_up'])
lumiSections_range = [LumiSectionsRange_low, LumiSectionsRange_up]
HLT_name = config['HLT']['HLTname']

N_den = 0
N_num = 0
for i in tqdm(range(number_of_ephemeral_folder)):
    f = open(path_json + "folder_" + str(i) +".json")
    data = json.load(f)
    for file in data.keys():
        N_den += data[file]['N_den']
        N_num += data[file]['N_num']
    f.close()

print(f"Total number of events belonging to run {RefRun} and in LumiSections range {lumiSections_range}: {N_den}")
print(f"... and passing {HLT_name}: {N_num}")

eff, err_low, err_up = compute_eff_witherr(N_num, N_den)

print(f"Eff : {eff*L1A_physics}")
print(f"Eff_up : {err_low*L1A_physics}")
print(f"Eff_down : {err_up*L1A_physics}")