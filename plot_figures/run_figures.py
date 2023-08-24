from helpers import delta_r
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_Jets(events):
    jets_dict = {"pt_corr": events["Jet_PNet_ptcorr"].compute()*events["Jet_pt"].compute(), "pt": events["Jet_pt"].compute(), "eta": events["Jet_eta"].compute(), "phi": events["Jet_phi"].compute()}
    Jets = ak.zip(jets_dict)
    return Jets

def get_GenTaus(events):
    gentaus_dict = {"pt": events["GenLepton_pt"].compute(), "eta": events["GenLepton_eta"].compute(), "phi": events["GenLepton_phi"].compute()}
    GenTaus = ak.zip(gentaus_dict)
    return GenTaus

def high_eta_GenTau_selection(events):
    # return mask for GenLepton for hadronic GenTau passing minimal selection
    mask_GenTau = (events['GenLepton_pt'].compute() >= 20) & (np.abs(events['GenLepton_eta'].compute()) >= 2) & (events['GenLepton_kind'].compute() == 5)
    return mask_GenTau

def hGenTau_selection(events):
    # return mask for GenLepton for hadronic GenTau passing minimal selection
    mask_GenTau = (events['GenLepton_pt'].compute() >= 20) & (np.abs(events['GenLepton_eta'].compute()) <= 2.5) & (events['GenLepton_kind'].compute() == 5)
    return mask_GenTau

def matching_Jets_GenTaus(Jets, GenTaus, dR_matching_min = 0.5):
    # select thet pair Jet-Gentaus that match the best with min Dr <0.5
    GenTaus_inpair, Jets_inpair = ak.unzip(ak.cartesian([GenTaus, Jets], nested=True))  
    dR_GenTaus_Jets = delta_r(GenTaus_inpair, Jets_inpair)
    mask_GenTaus_Jets = (dR_GenTaus_Jets < dR_matching_min)
    mask = ak.any(mask_GenTaus_Jets, axis=-1) # return mask on GenTau if there is any match with a jet within dr<0.5 
    min_arg = ak.argmin(dR_GenTaus_Jets, axis=-1, keepdims = True , mask_identity = True) # return mask on obj_inpair where is the jet with min dr
    Gentau_sel = (GenTaus_inpair[min_arg])[mask]
    Jet_sel = (Jets_inpair[min_arg])[mask]
    return ak.flatten(Gentau_sel), ak.flatten(Jet_sel)

def plot_gaussian(x_data, plot_param, n_bins = 100, range_hist = [-0.2,0.2]):

    hist, bin_edges = np.histogram(x_data, bins=n_bins, range=range_hist)
    hist=hist/sum(hist)

    n = len(hist)
    x_hist=np.zeros((n),dtype=float) 
    for ii in range(n):
        x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
        
    y_hist=hist
        
    #Calculating the Gaussian PDF values given Gaussian parameters and random variable X
    def gaus(X,C,X_mean,sigma):
        return C*np.exp(-(X-X_mean)**2/(2*sigma**2))

    mean = sum(x_hist*y_hist)/sum(y_hist)                  
    sigma = sum(y_hist*(x_hist-mean)**2)/sum(y_hist) 

    #Gaussian least-square fitting process
    param_optimised,param_covariance_matrix = curve_fit(gaus,x_hist,y_hist,p0=[max(y_hist),mean,sigma],maxfev=5000)

    #PLOTTING THE GAUSSIAN CURVE -----------------------------------------
    fig = plt.figure()
    x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
    plt.plot(x_hist_2,gaus(x_hist_2,*param_optimised),'r.:',label='Gaussian fit')
    plt.plot([], [], ' ', label=f"Sigma: {round(param_optimised[2], 3)} +- {round(np.sqrt(param_covariance_matrix[2,2]),3)}")
    plt.plot([], [], ' ', label=f"Mean: {round(param_optimised[1], 3)} +- {round(np.sqrt(param_covariance_matrix[1,1]),3)}")
    plt.legend()

    #Normalise the histogram values
    weights = np.ones_like(x_data) / len(x_data)
    plt.hist(x_data, bins=n_bins, range=range_hist, weights=weights)

    #setting the label,title and grid of the plot
    plt.xlabel(plot_param['xlabel'])
    plt.xlim(range_hist[0],range_hist[1])
    plt.ylabel(plot_param['ylabel'])
    plt.grid("on")
    plt.savefig('plot_figures/figures/' + plot_param['savefig'])
    #plt.show()
    return

path_to_file = '/afs/cern.ch/work/p/pdebryas/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1/GluGluHToTauTau_M-125.root'
events = uproot.dask(path_to_file+':Events', library='ak')

GenTaus = get_GenTaus(events)
Jets = get_Jets(events)

# for eta
mask_GenTau = high_eta_GenTau_selection(events)
GenTaus_higheta = GenTaus[mask_GenTau]
Gentau_sel, Jet_sel = matching_Jets_GenTaus(Jets, GenTaus_higheta, dR_matching_min = 0.5)

plot_param_eta = {
    'xlabel': '(Gentau.eta - MatchJet.eta) for eta >=2',
    'ylabel': 'Probability',
    'savefig': 'resolution_eta_ge2.pdf',
}

plot_gaussian(np.array(Gentau_sel.eta - Jet_sel.eta), plot_param_eta, n_bins = 100, range_hist = [-0.2,0.2])

# for pt
mask_GenTau = hGenTau_selection(events)
GenTaus_norm = GenTaus[mask_GenTau]
Gentau_sel, Jet_sel = matching_Jets_GenTaus(Jets, GenTaus_norm, dR_matching_min = 0.5)

plot_param_pt = {
    'xlabel': '(Gentau.pt - MatchJet.pt)',
    'ylabel': 'Probability',
    'savefig': 'resolution_pt.pdf',
}

plot_gaussian(np.array(Gentau_sel.pt - Jet_sel.pt), plot_param_pt, n_bins = 100, range_hist = [-30,30])

# for pt_corr
plot_param_pt_corr = {
    'xlabel': '(Gentau.pt - MatchJet.ptcorr)',
    'ylabel': 'Probability',
    'savefig': 'resolution_pt_corr.pdf',
}

plot_gaussian(np.array(Gentau_sel.pt - Jet_sel.pt_corr), plot_param_pt_corr, n_bins = 100, range_hist = [-30,30])


bins = [20, 30, 40, 50, 70, 90, 110, 150, 250]

std_delta_pt = []
std_delta_err = []
mean_delta_pt = []
std_delta_ptcorr = []
mean_delta_ptcorr = []
bins_mid = []
bins_err = []
for i in range(len(bins)-1):
    mask = (Gentau_sel.pt >= bins[i]) & (Gentau_sel.pt < bins[i+1])
    delta_pt = np.array((Gentau_sel.pt - Jet_sel.pt)/Gentau_sel.pt)[mask]
    delta_ptcorr = np.array((Gentau_sel.pt - Jet_sel.pt_corr)/Gentau_sel.pt)[mask]
    std_delta_err.append(1/(2*np.sqrt(np.sum(mask))))
    std_delta_pt.append(np.std(delta_pt))
    mean_delta_pt.append(np.mean(delta_pt))
    std_delta_ptcorr.append(np.std(delta_ptcorr))
    mean_delta_ptcorr.append(np.mean(delta_ptcorr))
    bins_mid.append((bins[i]+bins[i+1])/2)
    bins_err.append((bins[i+1]-bins[i])/2)

plt.errorbar(bins_mid, std_delta_pt, xerr=bins_err, yerr=std_delta_err, marker='+', label='pt', ls='none')
plt.errorbar(bins_mid, std_delta_ptcorr, xerr=bins_err, yerr=std_delta_err, marker='+', label='pt corr', ls='none')
plt.xlabel('GenTau vis pt', fontsize = 17)
plt.xlim(bins[0],bins[-1])
plt.ylabel(r'std($(p_T^{true}-p_T^{Jet})/p_T^{true}$)', fontsize = 17)
plt.grid("on")
plt.legend(prop={'size': 17})
plt.savefig('plot_figures/figures/std_pt_ptcorr.pdf')
plt.close()