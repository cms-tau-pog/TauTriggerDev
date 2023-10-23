import os
from plot_figures.helpers import get_eff_rel_improvment, get_abs_efficiency
import matplotlib.pyplot as plt
from plot_figures.params import WP_params_SingleTau, WP_params_DiTau, HLTnameDiTau, HLTnameSingleTau

#path
output_fig = f'{os.getenv("RUN_PATH")}/plot_figures/figures/'
data_eff_path =f'{os.getenv("RUN_PATH")}/ComputeEfficiency/result/'

sample_list = ['GluGluHToTauTau_M-125_ext1', 'VBFHToTauTau_M125', 'GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00', 'VBFHHto2B2Tau_CV-1_C2V-1_C3-1']

#-------------------------------------------------------------------------------------------------------------------------------------------------

#print relative improvment
print('Compute rel improvment for SingleTau:')
data_eff_path_SingleTau = data_eff_path + HLTnameSingleTau +'/'
rel_improvment_dict = {}
abs_efficiency_dict = {}

for WP in WP_params_SingleTau.keys():
    print(f'{WP} WP:')
    rel_improvment_dict[WP]={}
    abs_efficiency_dict[WP]={}

    for sample_av in sample_list:
        print(f'    - for {sample_av}:')

        fileName_dict = {
                        'DeepTau': [
                                    f"{data_eff_path_SingleTau}{HLTnameSingleTau}/{sample_av}.root"
                                    ],
                        'PNet': [
                                    f"{data_eff_path_SingleTau}PNetTresh_{WP_params_SingleTau[WP]['t1']}_{WP_params_SingleTau[WP]['t2']}_130/{sample_av}.root"
                                    ]
        }
        rel_improvment, rel_improvment_err = get_eff_rel_improvment(fileName_dict, n_min=1)
        rel_improvment_dict[WP][sample_av] = [rel_improvment, rel_improvment_err]
        print(f'          {round(rel_improvment, 3)} +- {round(rel_improvment_err, 3)}')

        eff, err_eff = get_abs_efficiency(fileName_dict['PNet'][0], n_min=1)
        abs_efficiency_dict[WP][sample_av] = [eff, err_eff]
        print(f'          (with PNet abs. eff.: {round(eff,4)} +- {round(err_eff,4)})')

print('DeepTau eff:')
for sample_av in sample_list:
    print(f'    - for {sample_av}:')
    eff, err_eff = get_abs_efficiency(f"{data_eff_path_SingleTau}{HLTnameSingleTau}/{sample_av}.root", n_min=1)
    abs_efficiency_dict[WP][sample_av] = [eff, err_eff]
    print(f'          abs. eff.: {round(eff,4)} +- {round(err_eff,4)}')

#plot relative improvment as function of rate
for sample_av in sample_list:
    rel_eff_vec = []
    rate_SingleTauWP = []
    for WP in WP_params_SingleTau.keys():
        rel_eff_vec.append(rel_improvment_dict[WP][sample_av][0])
        rate_SingleTauWP.append(WP_params_SingleTau[WP]['rate'])
    plt.plot(rate_SingleTauWP, rel_eff_vec, '+-', label=sample_av)
plt.legend()
plt.xscale('log')
plt.xlabel('Rate [Hz]')
plt.grid()
plt.ylabel('eff. PNet / eff. DeepTau')
plt.savefig(output_fig + '/rate_eff_SingleTau.pdf')
plt.close()


#print relative improvment
print('')
print('Compute rel improvment for DiTau:')
data_eff_path_DiTau = data_eff_path + HLTnameDiTau +'/'
rel_improvment_dict = {}
abs_efficiency_dict = {}

for WP in WP_params_DiTau.keys():
    print(f'{WP} WP:')
    rel_improvment_dict[WP]={}
    abs_efficiency_dict[WP]={}

    for sample_av in sample_list:
        print(f'    - for {sample_av}:')

        fileName_dict = {
                        'DeepTau': [
                                    f"{data_eff_path_DiTau}{HLTnameDiTau}/{sample_av}.root"
                                    ],
                        'PNet': [
                                    f"{data_eff_path_DiTau}PNetTresh_{WP_params_DiTau[WP]['t1']}_{WP_params_DiTau[WP]['t2']}_30/{sample_av}.root"
                                    ]
        }
        rel_improvment, rel_improvment_err = get_eff_rel_improvment(fileName_dict, n_min=2)
        rel_improvment_dict[WP][sample_av] = [rel_improvment, rel_improvment_err]
        print(f'          {round(rel_improvment, 3)} +- {round(rel_improvment_err, 3)}')

        eff, err_eff = get_abs_efficiency(fileName_dict['PNet'][0], n_min=2)
        abs_efficiency_dict[WP][sample_av] = [eff, err_eff]
        print(f'          (with PNet abs. eff.: {round(eff,4)} +- {round(err_eff,4)})')

print('DeepTau eff:')
for sample_av in sample_list:
    print(f'    - for {sample_av}:')
    eff, err_eff = get_abs_efficiency(f"{data_eff_path_DiTau}{HLTnameDiTau}/{sample_av}.root", n_min=2)
    abs_efficiency_dict[WP][sample_av] = [eff, err_eff]
    print(f'          abs. eff.: {round(eff,4)} +- {round(err_eff,4)}')

#plot relative improvment as function of rate
for sample_av in sample_list:
    rel_eff_vec = []
    rate_DiTauWP = []
    for WP in WP_params_DiTau.keys():
        rel_eff_vec.append(rel_improvment_dict[WP][sample_av][0])
        rate_DiTauWP.append(WP_params_DiTau[WP]['rate'])
    plt.plot(rate_DiTauWP, rel_eff_vec, '+-', label=sample_av)
plt.legend()
plt.xscale('log')
plt.xlabel('Rate [Hz]')
plt.grid()
plt.ylabel('eff. PNet / eff. DeepTau')
plt.savefig(output_fig + '/rate_eff_DoubleTau.pdf')
plt.close()