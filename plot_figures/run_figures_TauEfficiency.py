import os
import numpy as np
from plot_figures.helpers import get_eff_rel_improvment, get_abs_efficiency, plot_eff_tau
import matplotlib.pyplot as plt
from plot_figures.params import WP_params_SingleTau, WP_params_DiTau, rate_deepTau_DiTau, rate_deepTau_SingleTau, HLTnameDiTau, HLTnameSingleTau

#param
output_fig = f'{os.getenv("RUN_PATH")}/plot_figures/figures/'
data_eff_path =f'{os.getenv("RUN_PATH")}/ComputeEfficiency/result/'

#sample_list = ['ZprimeToTauTau_M-4000', 'VBFHToTauTau_M125', 'GluGluHToTauTau_M-125',  'VBFHHto2B2Tau_CV-1_C2V-1_C3-1']
sample_list = ['ZprimeToTauTau_M-4000', 'VBFHToTauTau_M125',  'VBFHHto2B2Tau_CV-1_C2V-1_C3-1', 'GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00', 'GluGluHToTauTau_M-125', 'GluGluHToTauTau_M-125_ext1']


#-------------------------------------------------------------------------------------------------------------------------------------------------
#figures Tau eff
print('')
print('Figures Tau eff SingleTau:')

bin_dict = {
    'pt': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 330, 360,400, 450, 500, 700, 1000, 1500, 2000, 3000],
    'eta': [-2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
    'phi': [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
}
var_list = ['pt','eta','phi']

data_eff_path_SingleTau = data_eff_path + HLTnameSingleTau +'/'
fileName_dict = {}
for WP in WP_params_SingleTau.keys():
    rate = round(WP_params_SingleTau[WP]['rate'],0)
    list_filename = []
    for sample_av in sample_list:
        filename = f'{data_eff_path_SingleTau}PNetTresh_{WP_params_SingleTau[WP]["t1"]}_{WP_params_SingleTau[WP]["t2"]}_130/{sample_av}.root'
        list_filename.append(filename)
    fileName_dict[f'{WP}_{str(rate)}Hz'] = list_filename


rate_deepTau = round(rate_deepTau_SingleTau,0)
list_filename = []
for sample_av in sample_list:
    filename = f"{data_eff_path_SingleTau}{HLTnameSingleTau}/{sample_av}.root"
    list_filename.append(filename)
fileName_dict[f'DeepTau_{str(rate_deepTau)}Hz'] = list_filename

for var in var_list:
    savefig_path = output_fig + f'SingleTau/SingleTau_{var}.pdf'
    plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = None, Hadron_mask = None) 
    savefig_path = output_fig +f'SingleTau/SingleTau_ptau_{var}.pdf'
    plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = 1, Hadron_mask = None) 
    savefig_path = output_fig +f'SingleTau/SingleTau_ntau_{var}.pdf'
    plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = -1, Hadron_mask = None) 
    for i in [1,2,3,4]:
        savefig_path = output_fig +f'SingleTau/SingleTau_HadronCfg{i}_{var}.pdf'
        plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = None, Hadron_mask = i) 
    for i in [1,2,3,4]:
        savefig_path = output_fig +f'SingleTau/SingleTau_PU{i}_{var}.pdf'
        plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = None, Hadron_mask = None, PU_mask=i) 


print('')
print('Figures Tau eff DiTau:')

bin_dict = {
    'pt': [20, 25, 30, 35, 40, 45, 50, 60,70,80,90,100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 330, 360,400, 450, 500, 700, 1000, 1500, 2000, 3000],
    'eta': [-2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
    'phi': [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
}
var_list = ['pt','eta','phi']

data_eff_path_DiTau = data_eff_path + HLTnameDiTau +'/'
fileName_dict = {}
for WP in WP_params_DiTau.keys():
    rate = round(WP_params_DiTau[WP]['rate'],0)
    list_filename = []
    for sample_av in sample_list:
        filename = f'{data_eff_path_DiTau}PNetTresh_{WP_params_DiTau[WP]["t1"]}_{WP_params_DiTau[WP]["t2"]}_30/{sample_av}.root'
        list_filename.append(filename)
    fileName_dict[f'{WP}_{str(rate)}Hz'] = list_filename


rate_deepTau = round(rate_deepTau_DiTau,0)
list_filename = []
for sample_av in sample_list:
    filename = f"{data_eff_path_DiTau}{HLTnameDiTau}/{sample_av}.root"
    list_filename.append(filename)
fileName_dict[f'DeepTau_{str(rate_deepTau)}Hz'] = list_filename

for var in var_list:
    savefig_path = output_fig + f'DoubleTau/DoubleTau_{var}.pdf'
    plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = None, Hadron_mask = None) 
    savefig_path = output_fig +f'DoubleTau/DoubleTau_ptau_{var}.pdf'
    plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = 1, Hadron_mask = None) 
    savefig_path = output_fig +f'DoubleTau/DoubleTau_ntau_{var}.pdf'
    plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = -1, Hadron_mask = None) 
    for i in [1,2,3,4]:
        savefig_path = output_fig +f'DoubleTau/DoubleTau_HadronCfg{i}_{var}.pdf'
        plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = None, Hadron_mask = i) 
    for i in [1,2,3,4]:
        savefig_path = output_fig +f'DoubleTau/DoubleTau_PU{i}_{var}.pdf'
        plot_eff_tau(fileName_dict, bin_dict[var], savefig_path, var, mask_charge = None, Hadron_mask = None, PU_mask=i)


