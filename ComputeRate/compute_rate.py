import json
from helpers import compute_eff_witherr, load_cfg_file_ComputeRate
from tqdm import tqdm

config = load_cfg_file_ComputeRate()
number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
output_path = config['DATA']['output_path']
L1A_physics = float(config['RUNINFO']['L1A_physics'])
RefRun = int(config['RUNINFO']['ref_run'])
LumiSectionsRange_low = int(config['RUNINFO']['LumiSectionsRange_low'])
LumiSectionsRange_up = int(config['RUNINFO']['LumiSectionsRange_up'])
lumiSections_range = [LumiSectionsRange_low, LumiSectionsRange_up]
HLT_name = config['HLT']['HLTname']
L1_name = config['HLT']['L1name']
mode = config['MODE']['Rate_to_compute']
HLT_rate = config['HLT']['HLT_rate']
L1_rate = config['HLT']['L1_rate']

path_result = output_path + f'result_{RefRun}_' + mode+ '/'

N_den = 0
N_num = 0
for i in tqdm(range(number_of_ephemeral_folder)):
    f = open(path_result + "folder_" + str(i) +".json")
    data = json.load(f)
    for file in data.keys():
        N_den += data[file]['N_den']
        N_num += data[file]['N_num']
    f.close()

eff, err_low, err_up = compute_eff_witherr(N_num, N_den)

print(f"Total number of events belonging to run {RefRun} and in LumiSections range {lumiSections_range}: {N_den}")

if mode == 'HLTflag':
    print(f"... and passing {HLT_name}: {N_num}")
    print(f"HLT rate in cms oms: {HLT_rate}")
if mode == 'HLTlogic':
    print(f"... and passing {HLT_name} conditions: {N_num}")
    print(f"HLT rate in cms oms: {HLT_rate}")
if mode == 'L1lflag':
    print(f"... and passing {L1_name}: {N_num}")
    print(f"L1 rate in cms oms: {L1_rate}")
if mode == 'L1logic':
    print(f"... and passing {L1_name} conditions: {N_num}")
    print(f"L1 rate in cms oms: {L1_rate}")

print('Computed rate: ')
print(f"Eff : {eff*L1A_physics}")
print(f"Eff_up : {err_low*L1A_physics}")
print(f"Eff_down : {err_up*L1A_physics}")