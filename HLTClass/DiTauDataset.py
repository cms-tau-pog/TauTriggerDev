import awkward as ak
import numpy as np
from HLTClass.dataset import Dataset
from HLTClass.dataset import get_L1Taus, get_Taus, get_Jets, get_GenTaus, hGenTau_selection, matching_Gentaus, matching_L1Taus_obj, compute_PNet_charge_prob

# ------------------------------ functions for DiTau with PNet -----------------------------------------------------------------------------
def compute_PNet_WP_DiTau(tau_pt, par):
    # return PNet WP for DiTau (to optimize)
    t1 = par[0]
    t2 = par[1]
    t3 = 0.001
    t4 = 0
    x1 = 30
    x2 = 100
    x3 = 500
    x4 = 1000
    PNet_WP = tau_pt*0.
    ones = tau_pt/tau_pt
    PNet_WP = ak.where((tau_pt <= ones*x1) == False, PNet_WP, ones*t1)
    PNet_WP = ak.where((tau_pt >= ones*x4) == False, PNet_WP, ones*t4)
    PNet_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, PNet_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
    PNet_WP = ak.where(((tau_pt >= ones*x2) & (tau_pt < ones*x3))== False, PNet_WP, (t3 - t2) / (x3 - x2) * (tau_pt - ones*x2) + ones*t2)
    PNet_WP = ak.where(((tau_pt >= ones*x3) & (tau_pt < ones*x4))== False, PNet_WP, (t4 - t3) / (x4 - x3) * (tau_pt - ones*x3) + ones*t3)
    return PNet_WP

def Jet_selection_DiTau(events, par, apply_PNET_WP = True):
    # return mask for Jet passing selection for DiTau path
    Jet_pt_corr = events['Jet_pt'].compute()*events['Jet_PNet_ptcorr'].compute()
    Jets_mask = (events['Jet_pt'].compute() >= 30) & (np.abs(events['Jet_eta'].compute()) <= 2.3) & (Jet_pt_corr >= 30)
    if apply_PNET_WP:
        probTauP = events['Jet_PNet_probtauhp'].compute()
        probTauM = events['Jet_PNet_probtauhm'].compute()
        Jets_mask = Jets_mask & ((probTauP + probTauM) >= compute_PNet_WP_DiTau(Jet_pt_corr, par)) & (compute_PNet_charge_prob(probTauP, probTauM) >= 0)
    return Jets_mask

def evt_sel_DiTau(events, par, n_min=2, is_gen = False):
    # Selection of event passing condition of DiTau with PNet HLT path + mask of objects passing those conditions

    L1Tau_IsoTau34er2p1L2NN_mask = L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
    L1Tau_Tau70er2p1L2NN_mask = L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
    DiTau_mask = Jet_selection_DiTau(events, par, apply_PNET_WP = True)
    # at least n_min L1tau/ recoJet should pass
    DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= n_min) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= n_min)) & (ak.sum(DiTau_mask, axis=-1) >= n_min)
    if is_gen:
        # if MC data, at least n_min GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        DiTau_evt_mask = DiTau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= n_min)

    # matching
    L1Taus = get_L1Taus(events)
    Jets = get_Jets(events)
    L1Taus_DoubleTau = get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1L2NN_mask, L1Tau_Tau70er2p1L2NN_mask, n_min_taus = n_min)
    Jets_DoubleTau = Jets[DiTau_mask]
    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_DoubleTau  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Jets_DoubleTau, GenTaus_DoubleTau)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)
    else:
        matchingJets_mask = matching_L1Taus_obj(L1Taus_DoubleTau, Jets_DoubleTau)
        # at least n_min Tau should match L1Tau
        evt_mask_matching = (ak.sum(matchingJets_mask, axis=-1) >= n_min)

    DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching
    if is_gen: 
        return DiTau_evt_mask, matchingGentaus_mask
    else:
        return DiTau_evt_mask, matchingJets_mask
    
# ------------------------------ functions for DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 ---------------------------------------------------
def compute_DeepTau_WP_DiTau(tau_pt):
    # return DeepTau WP for DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1
    t1 = 0.649
    t2 = 0.441
    t3 = 0.05
    x1 = 35
    x2 = 100
    x3 = 300
    Tau_WP = tau_pt*0.
    ones = tau_pt/tau_pt
    Tau_WP = ak.where((tau_pt <= ones*x1) == False, Tau_WP, ones*t1)
    Tau_WP = ak.where((tau_pt >= ones*x3) == False, Tau_WP, ones*t3)
    Tau_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, Tau_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
    Tau_WP = ak.where(((tau_pt >= ones*x2) & (tau_pt < ones*x3))== False, Tau_WP, (t3 - t2) / (x3 - x2) * (tau_pt - ones*x2) + ones*t2)
    return Tau_WP

def Tau_selection_DiTau(events, apply_DeepTau_WP = True):
    # return mask for Tau passing selection for DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1
    tau_pt = events['Tau_pt'].compute()
    Tau_mask = (tau_pt >= 35) & (np.abs(events['Tau_eta'].compute()) <= 2.1)
    if apply_DeepTau_WP:
        Tau_mask = Tau_mask & (events['Tau_deepTauVSjet'].compute() >= compute_DeepTau_WP_DiTau(tau_pt))
    return Tau_mask

def evt_sel_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(events, n_min = 2, is_gen = False):
    # Selection of event passing condition of DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 + mask of objects passing those conditions

    L1Tau_IsoTau34er2p1L2NN_mask = L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
    L1Tau_Tau70er2p1L2NN_mask = L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
    DiTau_mask = Tau_selection_DiTau(events)
    # at least n_min L1tau/ recoTau should pass
    DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= n_min) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= n_min)) & (ak.sum(DiTau_mask, axis=-1) >= n_min)
    if is_gen:
        # if MC data, at least n_min GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        DiTau_evt_mask = DiTau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= n_min)

    # matching
    L1Taus = get_L1Taus(events)
    Taus = get_Taus(events)
    L1Taus_DoubleTau = get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1L2NN_mask, L1Tau_Tau70er2p1L2NN_mask, n_min_taus = n_min)
    Taus_DoubleTau = Taus[DiTau_mask]
    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_DoubleTau  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Taus_DoubleTau, GenTaus_DoubleTau)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)
    else:
        matchingTaus_mask = matching_L1Taus_obj(L1Taus_DoubleTau, Taus_DoubleTau)
        # at least n_min Tau should match L1Tau
        evt_mask_matching = (ak.sum(matchingTaus_mask, axis=-1) >= n_min)
    
    DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching
    if is_gen: 
        return DiTau_evt_mask, matchingGentaus_mask
    else:
        return DiTau_evt_mask, matchingTaus_mask

# ------------------------------ Common functions for Ditau path ---------------------------------------------------------------
def L1Tau_IsoTau34er2p1_selection(events):
    # return mask for L1tau passing IsoTau34er2p1 selection
    L1_IsoTau34er2p1_mask = (events['L1Tau_hwPt'].compute() >= 0x44) & (events['L1Tau_hwEta'].compute() <= 0x30) & (events['L1Tau_hwEta'].compute() >= -49) & (events['L1Tau_hwIso'].compute() > 0 )
    return L1_IsoTau34er2p1_mask

def L1Tau_Tau70er2p1_selection(events):
    # return mask for L1tau passing Tau70er2p1 selection
    L1_Tau70er2p1_mask  = (events['L1Tau_pt'].compute() >= 70) & (np.abs(events['L1Tau_eta'].compute() <= 2.131))
    return L1_Tau70er2p1_mask

def L1Tau_L2NN_selection_DiTau(events):
    # return mask for L1tau passing L2NN selection for DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1
    L1_L2NN_mask = ((events['L1Tau_l2Tag'].compute() > 0.386) | (events['L1Tau_pt'].compute() >= 250))
    return L1_L2NN_mask

def Denominator_Selection_DiTau(GenLepton):
    # return mask for event passing minimal Gen requirements for diTau HLT (2 hadronic Taus with min vis. pt and eta)
    mask = (GenLepton['kind'] == 5)
    ev_mask = ak.sum(mask, axis=-1) >= 2  # at least 2 Gen taus should pass this requirements
    return ev_mask

def get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = 2):
    # return L1tau that pass DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 selection (OR between L1Tau_IsoTau34er2p1 and L1Tau_Tau70er2p1)
    IsoTau34er2p1_and_Tau70er2p1 = (ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_taus) & (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_taus)
    IsoTau34er2p1_and_notTau70er2p1 = (ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_taus) & np.invert((ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_taus))
    notIsoTau34er2p1_and_Tau70er2p1 = np.invert((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_taus)) & (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_taus)
    L1Taus_Sel = L1Taus
    L1Taus_Sel = ak.where(IsoTau34er2p1_and_Tau70er2p1 == False, L1Taus_Sel, L1Taus[L1Tau_IsoTau34er2p1_mask | L1Tau_Tau70er2p1_mask])
    L1Taus_Sel = ak.where(IsoTau34er2p1_and_notTau70er2p1 == False, L1Taus_Sel, L1Taus[L1Tau_IsoTau34er2p1_mask]) 
    L1Taus_Sel = ak.where(notIsoTau34er2p1_and_Tau70er2p1 == False, L1Taus_Sel, L1Taus[L1Tau_Tau70er2p1_mask]) 
    return L1Taus_Sel
    




class DiTauDataset(Dataset):
    def __init__(self, fileName):
        Dataset.__init__(self, fileName)

    # ------------------------------ functions to Compute Rate ---------------------------------------------------------------------

    def get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(self):
        print(f'Computing rate for HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {len(events)}")

        DiTau_evt_mask, matchingTaus_mask = evt_sel_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(events, n_min = 2, is_gen = False)
        N_num = len(events[DiTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def get_Nnum_Nden_DiTauPNet(self, par):
        print(f'Computing Rate for DiTau path with param: {par}')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {len(events)}")

        DiTau_evt_mask, matchingJets_mask = evt_sel_DiTau(events, par, n_min=2, is_gen = False)
        N_num = len(events[DiTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    # ------------------------------ functions to Compute Efficiency ---------------------------------------------------------------

    def save_Event_Nden_eff_DiTau(self, tmp_file):
        #Save only needed informations for events passing minimal Gen requirements for diTau HLT (passing denominator selection for efficiency)
        events = self.get_events()
        print(f"Number of events in the file: {len(events)}")
        GenLepton = self.get_GenLepton(events)
        evt_mask = Denominator_Selection_DiTau(GenLepton)
        print(f"Number of events with at least 2 hadronic Tau (kind= TauDecayedToHadrons): {ak.sum(evt_mask)}")
        self.Save_Event_Nden_Eff(events, GenLepton, evt_mask, tmp_file)
        return

    def produceRoot_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(self, out_file):
        #load all events that pass denominator Selection
        events = self.get_events()
        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]
        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        DiTau_evt_mask, matchingGentaus_mask = evt_sel_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(events, n_min = 1, is_gen = True)
        Tau_Num = (Tau_Den[matchingGentaus_mask])[DiTau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")
        events = events[DiTau_evt_mask]

        self.save_info(events, Tau_Den, Tau_Num, out_file)
        return

    def produceRoot_DiTauPNet(self, out_file, par):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]
        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        DiTau_evt_mask, matchingGentaus_mask = evt_sel_DiTau(events, par, n_min=1, is_gen = True)

        Tau_Num = (Tau_Den[matchingGentaus_mask])[DiTau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")
        events = events[DiTau_evt_mask]

        self.save_info(events, Tau_Den, Tau_Num, out_file)

        return

    # ------------------------------ functions to Compute Efficiency for opt ---------------------------------------------------------------

    def ComputeEffAlgo_DiTauPNet(self, par):

        #load all events that pass denominator Selection
        events = self.get_events()

        L1Taus = get_L1Taus(events)
        Jets = get_Jets(events)
        GenTaus = get_GenTaus(events)
        n_min = 2

        # Selection of L1/Gen and Jets objects without PNET WP
        L1Tau_IsoTau34er2p1L2NN_mask = L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
        L1Tau_Tau70er2p1L2NN_mask = L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
        DiTau_mask = Jet_selection_DiTau(events, par, apply_PNET_WP = False)
        GenTau_mask = hGenTau_selection(events)
        # at least n_min L1tau/ recoJet should pass
        DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= n_min) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= n_min)) & (ak.sum(DiTau_mask, axis=-1) >= n_min) & (ak.sum(GenTau_mask, axis=-1) >= n_min)

        # matching
        L1Taus_DoubleTau = get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1L2NN_mask, L1Tau_Tau70er2p1L2NN_mask, n_min_taus = n_min)
        Jets_DoubleTau = Jets[DiTau_mask]
        GenTaus_DoubleTau  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Jets_DoubleTau, GenTaus_DoubleTau)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching
        N_den = len(events[DiTau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with PNET WP
        DiTau_mask = Jet_selection_DiTau(events, par, apply_PNET_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= n_min) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= n_min)) & (ak.sum(DiTau_mask, axis=-1) >= n_min) & (ak.sum(GenTau_mask, axis=-1) >= n_min)

        # matching
        Jets_DoubleTau = Jets[DiTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Jets_DoubleTau, GenTaus_DoubleTau)
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching
        N_num = len(events[DiTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def ComputeEffAlgo_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(self):

        #load all events that pass denominator Selection
        events = self.get_events()

        L1Taus = get_L1Taus(events)
        Taus = get_Taus(events)
        GenTaus = get_GenTaus(events)
        n_min = 2

        # Selection of L1/Gen and Jets objects without PNET WP
        L1Tau_IsoTau34er2p1L2NN_mask = L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
        L1Tau_Tau70er2p1L2NN_mask = L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
        DiTau_mask = Tau_selection_DiTau(events, apply_DeepTau_WP = False)
        GenTau_mask = hGenTau_selection(events)
        # at least n_min L1tau/ recoJet should pass
        DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= n_min) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= n_min)) & (ak.sum(DiTau_mask, axis=-1) >= n_min) & (ak.sum(GenTau_mask, axis=-1) >= n_min)

        # matching
        L1Taus_DoubleTau = get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1L2NN_mask, L1Tau_Tau70er2p1L2NN_mask, n_min_taus = n_min)
        Taus_DoubleTau = Taus[DiTau_mask]
        GenTaus_DoubleTau  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Taus_DoubleTau, GenTaus_DoubleTau)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching
        N_den = len(events[DiTau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with PNET WP
        DiTau_mask = Tau_selection_DiTau(events, apply_DeepTau_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= n_min) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= n_min)) & (ak.sum(DiTau_mask, axis=-1) >= n_min) & (ak.sum(GenTau_mask, axis=-1) >= n_min)

        # matching
        Taus_DoubleTau = Taus[DiTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Taus_DoubleTau, GenTaus_DoubleTau)
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching
        N_num = len(events[DiTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

