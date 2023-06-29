import os
import shutil
from statsmodels.stats.proportion import proportion_confint
import configparser
import numpy as np

def load_cfg_file():
    global_path = os.getenv("RUN_PATH")
    config_file = os.path.join(global_path, 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def compute_eff_witherr(N_num, N_den):
    conf_int = proportion_confint(count=N_num, nobs=N_den, alpha=0.32, method='beta')
    eff = N_num / N_den
    err_low = eff - conf_int[0]
    err_up = conf_int[1] - eff
    return eff, err_low, err_up

# loade list of files in pathname
def files_from_path(pathname):
    FileNameList = []
    for file in os.listdir(pathname):
        FileNameList.append(os.path.join(pathname, file))
    return FileNameList

def hadd_anatuple(output_tmp_folder, output_root_file):

    ''' Hadd all root files in output_tmp_folder to produce one file (output_root_file)
    '''

    if not os.path.exists(output_tmp_folder):
        print(f'Careful: {output_tmp_folder} do not exist !!!!')
        os.makedirs(output_tmp_folder)

    if os.path.exists(output_root_file):
        raise f'File {output_root_file} exist already, cannot move files from {output_tmp_folder}'
        
    filelist = files_from_path(output_tmp_folder)
            
    if len(filelist) == 0:
        return

    if len(filelist) == 1:
        tmp_file = filelist[0]
        shutil.move(tmp_file, output_root_file)
        return

    if len(filelist) > 1:

        filelist_cmd = ''
        for file in filelist:
            filelist_cmd = filelist_cmd + file + ' '

        hadd_cmd = 'hadd -n 11 ' + output_root_file + ' ' + filelist_cmd 
        print('Running the folowing command:')
        print(hadd_cmd)
        os.system(hadd_cmd)
 
    return

def delta_r2(v1, v2):
    dphi = (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi
    deta = v1.eta - v2.eta
    dr2 = dphi ** 2 + deta ** 2
    return dr2

def delta_r(v1, v2):
    return np.sqrt(delta_r2(v1, v2))