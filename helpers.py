import os
from statsmodels.stats.proportion import proportion_confint
import configparser

def load_cfg_file_ComputeRate():
    global_path = os.getenv("RUN_PATH")
    config_file = os.path.join(global_path, 'ComputeRate', 'config.ini')
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
        FileNameList.append(pathname+file)
    return FileNameList