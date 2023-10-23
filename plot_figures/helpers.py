import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

def compute_eff_witherr(N_num, N_den):
    conf_int = proportion_confint(count=N_num, nobs=N_den, alpha=0.32, method='beta')
    eff = N_num / N_den
    err_low = eff - conf_int[0]
    err_up = conf_int[1] - eff
    return eff, err_low, err_up

def compute_eff_witherr_vec(N_num, N_den):
    #compute efficiency giving N_num, N_den. 
    #return err with up and down value for unc.
    if len(N_num) != len(N_den):
        raise('error')
    eff = []
    err_low = []
    err_up = []
    for i in range(len(N_den)):
        if N_den[i] == 0.:
            eff.append(0)
            err_low.append(0)
            err_up.append(0)
        else:
            conf_int = proportion_confint(count=N_num[i], nobs=N_den[i], alpha=0.32, method='beta')
            div = N_num[i] / N_den[i]
            err_low.append(div - conf_int[0])
            err_up.append( conf_int[1] - div)
            eff.append(N_num[i] / N_den[i])
    return np.array(eff), np.array(err_low), np.array(err_up)

def plot_eff_tau(fileName_dict, bins, path_savefig, var, mask_charge = None, Hadron_mask = None, PU_mask = None):
    # Hadron_mask can be:
    # None --> no Hadron mask
    # 1 --> 1 nChargedHad, 0 nNeutralHad
    # 2 --> 1 nChargedHad, 1 nNeutralHad
    # 3 --> 1 nChargedHad, nNeutralHad > 1
    # 4 --> 3 nChargedHad (DM 10/11)

    # mask_charge can be:
    # None --> no charge mask
    # 1 --> only tau +
    #-1 --> only tau -

    # PU_mask can be:
    # None --> no Pileup mask
    # 1 --> tau in event passing 1 nPFPrimaryVertex
    # 2 --> tau in event passing 2 nPFPrimaryVertex
    # 3 --> tau in event passing 3 nPFPrimaryVertex
    # 4 --> tau in event passing >= 4 nPFPrimaryVertex

    print(f'Plotting {var}')
    if mask_charge == 1:
        print(f'... for taus +')
    if mask_charge == -1:
        print(f'... for taus -')
    if Hadron_mask != None:
        print(f'... for hadronic mask mode {Hadron_mask}')
    if PU_mask != None:
        print(f'... for PU mask {PU_mask}')
    unit = {
        'pt':  '[GeV]',
        'eta': '[]',
        'phi': '[]'
    }

    # Load datas ###############
    Tau_var = {}
    for filenamelist in fileName_dict.keys():
        Tau_var[filenamelist] = {}
        Tau_var[filenamelist]['TauDen_'+var] = []
        Tau_var[filenamelist]['TauNum_'+var] = []
        for filename in fileName_dict[filenamelist]:
            uproot_file = uproot.open(filename)
            if (Hadron_mask == 1) & (mask_charge == None) & (PU_mask == None):
                mask_Den = (uproot_file['TausDen']['Tau_nChargedHad'].array() == 1) & (uproot_file['TausDen']['Tau_nNeutralHad'].array() == 0)
                mask_Num = (uproot_file['TausNum']['Tau_nChargedHad'].array() == 1) & (uproot_file['TausNum']['Tau_nNeutralHad'].array() == 0)
                Tau_var[filenamelist]['TauDen_'+var] = ak.concatenate([Tau_var[filenamelist]['TauDen_'+var], ak.flatten(uproot_file['TausDen']['Tau_'+var].array()[mask_Den])])
                Tau_var[filenamelist]['TauNum_'+var] = ak.concatenate([Tau_var[filenamelist]['TauNum_'+var], ak.flatten(uproot_file['TausNum']['Tau_'+var].array()[mask_Num])])
            if (Hadron_mask == 2) & (mask_charge == None) & (PU_mask == None):
                mask_Den = (uproot_file['TausDen']['Tau_nChargedHad'].array() == 1) & (uproot_file['TausDen']['Tau_nNeutralHad'].array() == 1)
                mask_Num = (uproot_file['TausNum']['Tau_nChargedHad'].array() == 1) & (uproot_file['TausNum']['Tau_nNeutralHad'].array() == 1)
                Tau_var[filenamelist]['TauDen_'+var] = ak.concatenate([Tau_var[filenamelist]['TauDen_'+var], ak.flatten(uproot_file['TausDen']['Tau_'+var].array()[mask_Den])])
                Tau_var[filenamelist]['TauNum_'+var] = ak.concatenate([Tau_var[filenamelist]['TauNum_'+var], ak.flatten(uproot_file['TausNum']['Tau_'+var].array()[mask_Num])])
            if (Hadron_mask == 3) & (mask_charge == None) & (PU_mask == None):
                mask_Den = (uproot_file['TausDen']['Tau_nChargedHad'].array() == 1) & (uproot_file['TausDen']['Tau_nNeutralHad'].array() > 1)
                mask_Num = (uproot_file['TausNum']['Tau_nChargedHad'].array() == 1) & (uproot_file['TausNum']['Tau_nNeutralHad'].array() > 1)
                Tau_var[filenamelist]['TauDen_'+var] = ak.concatenate([Tau_var[filenamelist]['TauDen_'+var], ak.flatten(uproot_file['TausDen']['Tau_'+var].array()[mask_Den])])
                Tau_var[filenamelist]['TauNum_'+var] = ak.concatenate([Tau_var[filenamelist]['TauNum_'+var], ak.flatten(uproot_file['TausNum']['Tau_'+var].array()[mask_Num])])
            if (Hadron_mask == 4) & (mask_charge == None) & (PU_mask == None):
                mask_Den = (uproot_file['TausDen']['Tau_nChargedHad'].array() == 3) #& (uproot_file['TausDen']['Tau_nNeutralHad'].array() == 0)
                mask_Num = (uproot_file['TausNum']['Tau_nChargedHad'].array() == 3) #& (uproot_file['TausNum']['Tau_nNeutralHad'].array() == 0)
                Tau_var[filenamelist]['TauDen_'+var] = ak.concatenate([Tau_var[filenamelist]['TauDen_'+var], ak.flatten(uproot_file['TausDen']['Tau_'+var].array()[mask_Den])])
                Tau_var[filenamelist]['TauNum_'+var] = ak.concatenate([Tau_var[filenamelist]['TauNum_'+var], ak.flatten(uproot_file['TausNum']['Tau_'+var].array()[mask_Num])])
            #if (Hadron_mask == 5) & (mask_charge == None) & (PU_mask == None):
            #    mask_Den = (uproot_file['TausDen']['Tau_nChargedHad'].array() == 3) & (uproot_file['TausDen']['Tau_nNeutralHad'].array() > 0)
            #    mask_Num = (uproot_file['TausNum']['Tau_nChargedHad'].array() == 3) & (uproot_file['TausNum']['Tau_nNeutralHad'].array() > 0)
            #    Tau_var[filenamelist]['TauDen_'+var] = ak.concatenate([Tau_var[filenamelist]['TauDen_'+var], ak.flatten(uproot_file['TausDen']['Tau_'+var].array()[mask_Den])])
            #    Tau_var[filenamelist]['TauNum_'+var] = ak.concatenate([Tau_var[filenamelist]['TauNum_'+var], ak.flatten(uproot_file['TausNum']['Tau_'+var].array()[mask_Num])])
            if (mask_charge != None) & (Hadron_mask == None) & (PU_mask == None):
                mask_Den = (uproot_file['TausDen']['Tau_charge'].array() == mask_charge)
                mask_Num = (uproot_file['TausNum']['Tau_charge'].array() == mask_charge)
                Tau_var[filenamelist]['TauDen_'+var] = ak.concatenate([Tau_var[filenamelist]['TauDen_'+var], ak.flatten(uproot_file['TausDen']['Tau_'+var].array()[mask_Den])])
                Tau_var[filenamelist]['TauNum_'+var] = ak.concatenate([Tau_var[filenamelist]['TauNum_'+var], ak.flatten(uproot_file['TausNum']['Tau_'+var].array()[mask_Num])])
            if (mask_charge == None) & (Hadron_mask == None) & (PU_mask != None):
                if PU_mask == 4:
                    mask_Den = (uproot_file['eventsDen']['nPFPrimaryVertex'].array() >= 4)
                    mask_Num = (uproot_file['eventsNum']['nPFPrimaryVertex'].array() >= 4)
                else:
                    mask_Den = (uproot_file['eventsDen']['nPFPrimaryVertex'].array() == PU_mask)
                    mask_Num = (uproot_file['eventsNum']['nPFPrimaryVertex'].array() == PU_mask)
                Tau_var[filenamelist]['TauDen_'+var] = ak.concatenate([Tau_var[filenamelist]['TauDen_'+var], ak.flatten(uproot_file['TausDen']['Tau_'+var].array()[mask_Den])])
                Tau_var[filenamelist]['TauNum_'+var] = ak.concatenate([Tau_var[filenamelist]['TauNum_'+var], ak.flatten(uproot_file['TausNum']['Tau_'+var].array()[mask_Num])])
            if (mask_charge == None) & (Hadron_mask == None)& (PU_mask == None):
                Tau_var[filenamelist]['TauDen_'+var] = ak.concatenate([Tau_var[filenamelist]['TauDen_'+var], ak.flatten(uproot_file['TausDen']['Tau_'+var].array())])
                Tau_var[filenamelist]['TauNum_'+var] = ak.concatenate([Tau_var[filenamelist]['TauNum_'+var], ak.flatten(uproot_file['TausNum']['Tau_'+var].array())])

    # Plot Tau_var distributions and compute ratio  #####################################
    ratio = {}
    for filename in fileName_dict.keys():
        if len(Tau_var[filename]['TauDen_'+var]) ==0:
            print('No taus with that mask')
        ratio[filename] = {}
        val_of_bins_Den, _, _ = plt.hist(Tau_var[filename]['TauDen_'+var], bins = bins, range=(bins[0],bins[-1]) )
        val_of_bins_Num, _, _ = plt.hist(Tau_var[filename]['TauNum_'+var], bins = bins, range=(bins[0],bins[-1]) )
        ratio[filename]['ratio'], ratio[filename]['err_low'], ratio[filename]['err_up'] = compute_eff_witherr_vec(val_of_bins_Num, val_of_bins_Den)
        #print(f'Mean eff {filename}: {np.mean(ratio[filename]["ratio"])}')
    plt.clf()

    # Final plot  #####################################
    bincenter = 0.5 * (np.array(bins)[1:] + np.array(bins)[:-1])
    binsize = (np.array(bins)[1:] - np.array(bins)[:-1])*0.5
    fig = plt.figure()
    for filename in fileName_dict.keys():
        plt.errorbar(bincenter, ratio[filename]['ratio'], xerr=binsize, yerr=[ratio[filename]['err_low'], ratio[filename]['err_up']], fmt='.', label=filename)

    plt.xlim(bins[0],bins[-1])
    if var == 'phi':
        if '/SingleTau/' in path_savefig:
            plt.ylim(0,0.6)
        if '/DoubleTau/' in path_savefig:
            plt.ylim(0,0.8)
    plt.xlabel("GenTau vis "+ var +' '+ unit[var], fontsize = 15)
    plt.ylabel("Efficiency", fontsize = 15)
    plt.legend(fontsize = 14)
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    if var == 'pt':
        plt.xscale('log')
        if '/SingleTau/' in path_savefig:
            plt.xticks([100, 150, 200, 500,1000,1500,3000], ['100', '150','200','500','1000','1500','3000'], fontsize = 15)
        if '/DoubleTau/' in path_savefig:
            plt.xticks([20, 50, 100, 200, 500,1000,3000], ['20', '50', '100','200','500','1000','3000'], fontsize = 15)
    if var == 'eta':
        plt.xticks([-2.3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.3], ['-2.3', '-2', '-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5', '2', '2.3'], fontsize = 15)
    plt.tight_layout()
    plt.savefig(path_savefig)
    plt.close()
    return

def get_eff_rel_improvment(fileName_dict, n_min = 2):
    N_den_DeepTau = 0
    N_num_DeepTau = 0
    N_den_PNet = 0
    N_num_PNet = 0

    for filenamelist in fileName_dict.keys():
        for filename in fileName_dict[filenamelist]:
            uproot_file = uproot.open(filename)
            if filenamelist == 'DeepTau':
                N_den_DeepTau += ak.sum(uproot_file['TausDen']['nTau_pt'].array() >= 2)
                N_num_DeepTau += ak.sum(uproot_file['TausNum']['nTau_pt'].array() >= n_min)
            else:
                N_den_PNet += ak.sum(uproot_file['TausDen']['nTau_pt'].array() >= 2)
                N_num_PNet += ak.sum(uproot_file['TausNum']['nTau_pt'].array() >= n_min)

    #eff_PNet = N_num_PNet/N_den_PNet
    eff_PNet, eff_PNet_errlow, eff_PNet_errup = compute_eff_witherr(N_num_PNet, N_den_PNet)
    eff_PNet_err = np.max([eff_PNet_errlow,eff_PNet_errup])
    #print(f'eff PNet: {eff_PNet}')
    #Eff_DeepTau = N_num_DeepTau/N_den_DeepTau
    Eff_DeepTau, Eff_DeepTau_errlow, Eff_DeepTau_errup = compute_eff_witherr(N_num_DeepTau, N_den_DeepTau)
    eff_DeepTau_err = np.max([Eff_DeepTau_errlow,Eff_DeepTau_errup])
    #print(f'eff DeepTau: {Eff_DeepTau}')
    rel_improvment = eff_PNet/Eff_DeepTau
    rel_improvment_err = rel_improvment*np.sqrt((eff_DeepTau_err/Eff_DeepTau)**2+(eff_PNet_err/eff_PNet)**2)
    return rel_improvment, rel_improvment_err

def get_abs_efficiency(filename, n_min = 2):
    uproot_file = uproot.open(filename)
    N_den = ak.sum(uproot_file['TausDen']['nTau_pt'].array() >= 2)
    N_num = ak.sum(uproot_file['TausNum']['nTau_pt'].array() >= n_min)
    eff, eff_errlow, eff_errup = compute_eff_witherr(N_num, N_den)
    err_eff = np.max([eff_errlow,eff_errup])
    return eff, err_eff

def get_abs_efficiency_mask(filepath, sample_list, nPU, n_min = 2):
    N_den = 0
    N_num = 0
    for sample_av in sample_list:
        filename = f"{filepath}{sample_av}.root"
        uproot_file = uproot.open(filename)
        mask_Den = (uproot_file['eventsDen']['nPFPrimaryVertex'].array() == nPU)
        mask_Num = (uproot_file['eventsNum']['nPFPrimaryVertex'].array() == nPU)
        N_den += ak.sum(uproot_file['TausDen']['nTau_pt'].array()[mask_Den] >= 2)
        N_num += ak.sum(uproot_file['TausNum']['nTau_pt'].array()[mask_Num] >= n_min)
    eff, eff_errlow, eff_errup = compute_eff_witherr(N_num, N_den)
    err_eff = np.max([eff_errlow,eff_errup])
    return eff, err_eff

def delta_r2(v1, v2):
    ''' Return (Delta r)^2 between two objects
    '''
    dphi = (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi
    deta = v1.eta - v2.eta
    dr2 = dphi ** 2 + deta ** 2
    return dr2

def delta_r(v1, v2):
    ''' Return Delta r between two objects
    '''
    return np.sqrt(delta_r2(v1, v2))