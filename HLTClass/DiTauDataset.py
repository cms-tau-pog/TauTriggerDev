import uproot
import awkward as ak
from HLTClass.dataset import Dataset
import numpy as np

class DiTauDataset(Dataset):
    def __init__(self, fileName):
        Dataset.__init__(self, fileName)

    def compute_DeepTau_WP_DiTau(self, tau_pt):
        # should return DeepTau WP 
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

    def compute_PNet_WP_DiTau(self, tau_pt, par):
        # should return PNet WP 
        t1 = 0.6
        t2 = par[0]
        t3 = 0.05
        x1 = 30
        x2 = par[1]
        x3 = 300
        PNet_WP = tau_pt*0.
        ones = tau_pt/tau_pt
        PNet_WP = ak.where((tau_pt <= ones*x1) == False, PNet_WP, ones*t1)
        PNet_WP = ak.where((tau_pt >= ones*x3) == False, PNet_WP, ones*t3)
        PNet_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, PNet_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
        PNet_WP = ak.where(((tau_pt >= ones*x2) & (tau_pt < ones*x3))== False, PNet_WP, (t3 - t2) / (x3 - x2) * (tau_pt - ones*x2) + ones*t2)
        return PNet_WP
    
    def L1Tau_IsoTau34er2p1_selection(self, events):
        L1_IsoTau34er2p1_mask = (events['L1Tau_hwPt'].compute() >= 0x44) & (events['L1Tau_hwEta'].compute() <= 0x30) & (events['L1Tau_hwEta'].compute() >= -49) & (events['L1Tau_hwIso'].compute() > 0 )
        return L1_IsoTau34er2p1_mask

    def L1Tau_Tau70er2p1_selection(self, events):
        L1_Tau70er2p1_mask  = (events['L1Tau_pt'].compute() >= 70) & (np.abs(events['L1Tau_eta'].compute() <= 2.1))
        return L1_Tau70er2p1_mask

    def L1Tau_L2NN_selection_DiTau(self, events):
        L2NN_mask = ((events['L1Tau_l2Tag'].compute() > 0.386) | (events['L1Tau_pt'].compute() >= 250))
        return L2NN_mask
    
    def Tau_selection_DiTau(self, events, apply_DeepTau_WP = True):
        # Apply reco Tau selection
        Tau_mask = (events['Tau_pt'].compute() >= 35) & (np.abs(events['Tau_eta'].compute()) <= 2.1)
        if apply_DeepTau_WP:
            Tau_mask = Tau_mask & (events['Tau_deepTauVSjet'].compute() >= self.compute_DeepTau_WP_DiTau(events['Tau_pt'].compute()))
        return Tau_mask
    
    def GenTau_selection_DiTau(self, events):
        # Minimal requirements so that event pass 
        GenTau_mask = (events['GenLepton_pt'].compute() >= 20) & (np.abs(events['GenLepton_eta'].compute()) <= 2.1) & (events['GenLepton_kind'].compute() == 5)
        return GenTau_mask

    def Jet_selection_DiTau(self, events, par, apply_PNET_WP = True):
        Jet_pt = events['Jet_pt'].compute()*events['Jet_PNet_ptcorr'].compute()
        Jets_mask = (events['Jet_pt'].compute() >= 30) & (np.abs(events['Jet_eta'].compute()) <= 2.5) & (Jet_pt >= 30)
        if apply_PNET_WP:
            probTauP = events['Jet_PNet_probtauhp'].compute()
            probTauM = events['Jet_PNet_probtauhm'].compute()
            Jets_mask = Jets_mask & ((probTauP + probTauM) >= self.compute_PNet_WP_DiTau(Jet_pt, par)) & (self.compute_PNet_charge_prob(probTauP, probTauM) >= par[2])
        return Jets_mask

    def get_selL1Taus(self, L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = 2):
        IsoTau34er2p1_and_Tau70er2p1 = (ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_taus) & (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_taus)
        IsoTau34er2p1_and_notTau70er2p1 = (ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_taus) & np.invert((ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_taus))
        notIsoTau34er2p1_and_Tau70er2p1 = np.invert((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_taus)) & (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_taus)

        L1Taus_Sel = L1Taus
        L1Taus_Sel = ak.where(IsoTau34er2p1_and_Tau70er2p1 == False, L1Taus_Sel, L1Taus[L1Tau_IsoTau34er2p1_mask | L1Tau_Tau70er2p1_mask])
        L1Taus_Sel = ak.where(IsoTau34er2p1_and_notTau70er2p1 == False, L1Taus_Sel, L1Taus[L1Tau_IsoTau34er2p1_mask]) 
        L1Taus_Sel = ak.where(notIsoTau34er2p1_and_Tau70er2p1 == False, L1Taus_Sel, L1Taus[L1Tau_Tau70er2p1_mask]) 
        return L1Taus_Sel
    
    def Denominator_Selection_DiTau(self, GenLepton):
        # For diTau HLT: 2 hadronic Gen Tau that match vis. pt and eta requirements
        mask = (GenLepton['pt'] >= 20) & (np.abs(GenLepton['eta']) <= 2.1) & (GenLepton['kind'] == 5)
        ev_mask = ak.sum(mask, axis=-1) >= 2  # at least 2 Gen taus should pass this requirements
        return ev_mask
    
# ------------------------------ functions for ComputeEfficiency ---------------------------------------------------------------
    def save_Event_Nden_eff_DiTau(self, tmp_file):
        ''' 
        Save only needed informations (for numerator cuts) of events passing denominator cuts (Denominator_Selection)
        '''
        events = self.get_events()
        print(f"Number of events in the file: {len(events)}")
        GenLepton = self.get_GenLepton(events)
        ev_mask = self.Denominator_Selection_DiTau(GenLepton)
        print(f"Number of events with at least 2 Gen Tau with vis. pt >= 20, |eta|<2.1 and kind=5 (TauDecayedToHadrons): {ak.sum(ev_mask)}")

        saved_info_events = ['Tau_pt', 
                             'Tau_eta', 
                             'Tau_phi', 
                             'Tau_deepTauVSjet', 
                             'L1Tau_pt', 
                             'L1Tau_eta', 
                             'L1Tau_phi', 
                             'L1Tau_hwPt', 
                             'L1Tau_hwEta', 
                             'L1Tau_hwIso', 
                             'L1Tau_l2Tag',
                             'Jet_PNet_probtauhm',
                             'Jet_PNet_probtauhp',
                             'Jet_PNet_ptcorr',
                             'Jet_pt',
                             'Jet_eta',
                             'Jet_phi']
        
        saved_info_GenLepton = ['pt', 
                                'eta', 
                                'phi', 
                                'kind']

        lst = {}
        for element in saved_info_events:
            lst[element] = (events[element].compute())[ev_mask]

        for element in saved_info_GenLepton:
            lst['GenLepton_' + element] = (GenLepton[element])[ev_mask]

        with uproot.create(tmp_file, compression=uproot.ZLIB(4)) as file:
            file["Events"] = lst
        return

    def produceRoot_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(self, out_file):
        n_min_tau = 1

        # Get events that pass Denominator_Selection
        events = self.get_events()
        print(f"Number of events passing denominator selection: {len(events)}")

        # Selection at L1, reco and Gen Level for HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1
        L1Tau_L2NN_mask = self.L1Tau_L2NN_selection_DiTau(events)
        L1Tau_IsoTau34er2p1_mask = self.L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_mask
        L1Tau_Tau70er2p1_mask = self.L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_mask
        GenTau_mask = self.GenTau_selection_DiTau(events)
        Tau_mask = self.Tau_selection_DiTau(events)

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        Tau_Den = self.get_GenTaus(events)
        Tau_Den = Tau_Den[GenTau_mask]

        evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_tau) | (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_tau)) & (ak.sum(GenTau_mask, axis=-1) >= n_min_tau) & (ak.sum(Tau_mask, axis=-1) >= n_min_tau)
        events = events[evt_mask]
        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        GenTau_mask = GenTau_mask[evt_mask]
        Tau_mask = Tau_mask[evt_mask]
        print(f"Number of events with at least {n_min_tau} Tau (L1Tau, GenTau and reco Tau) with HLT requirments: {len(events)}")

        L1Taus = self.get_L1Taus(events)
        GenTaus = self.get_GenTaus(events)
        Taus = self.get_Taus(events)

        L1Taus_Sel = self.get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = n_min_tau)
        GenTaus_Sel = GenTaus[GenTau_mask]
        Taus_Sel = Taus[Tau_mask]
        
        # matching
        matching_GenTaus_mask = self.matching_Gentaus(L1Taus_Sel, Taus_Sel, GenTaus_Sel)

        # select events where at least n Tau that match Gen, L1, Jets exist
        evt_mask = ak.sum(matching_GenTaus_mask, axis=-1) >= n_min_tau
        events = events[evt_mask]
        Tau_Num = (GenTaus[matching_GenTaus_mask])[evt_mask]

        print(f"Number of events after matching: {len(events)}")
        print('')
        self.save_info(Tau_Den, Tau_Num, out_file)

        return
    
    def produceRoot_DiTauPNet(self, out_file, par):
        n_min_tau = 1

        # Get events that pass denominator
        events = self.get_events()
        # N events in denominator passing Denominator_Selection
        print(f"Number of events in the denominator: {len(events)}")

        # Selection at L1, reco and Gen Level for DiTauPNet HLT
        L1Tau_L2NN_mask = self.L1Tau_L2NN_selection_DiTau(events)
        L1Tau_IsoTau34er2p1_mask = self.L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_mask
        L1Tau_Tau70er2p1_mask = self.L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_mask
        GenTau_mask = self.GenTau_selection_DiTau(events)
        Jet_mask = self.Jet_selection_DiTau(events, par)

        # To compute efficiency, we save in denominator GenTau which pass minimal selection 
        Tau_Den = self.get_GenTaus(events)
        Tau_Den = Tau_Den[GenTau_mask]

        evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_tau) | (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_tau)) & (ak.sum(GenTau_mask, axis=-1) >= n_min_tau) & (ak.sum(Jet_mask, axis=-1) >= n_min_tau) 
        events = events[evt_mask]
        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        GenTau_mask = GenTau_mask[evt_mask]
        Jet_mask = Jet_mask[evt_mask]
        print(f"Number of events with at least {n_min_tau} Tau (L1Tau, GenTau, and Jet) with cuts requirments: {len(events)}")

        L1Taus = self.get_L1Taus(events)
        GenTaus = self.get_GenTaus(events)
        Jets = self.get_Jets(events)

        L1Taus_Sel = self.get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = n_min_tau)
        GenTaus_Sel = GenTaus[GenTau_mask]
        Jets_Sel = Jets[Jet_mask]

        # matching
        matching_GenTaus_mask = self.matching_Gentaus(L1Taus_Sel, Jets_Sel, GenTaus_Sel)

        # select events where at least n Tau that match Gen, L1, Jets exist
        evt_mask = ak.sum(matching_GenTaus_mask, axis=-1) >= n_min_tau
        events = events[evt_mask]
        Tau_Num = (GenTaus[matching_GenTaus_mask])[evt_mask]

        print(f"Number of events after matching: {len(events)}")
        print('')
        self.save_info(Tau_Den, Tau_Num, out_file)

        return

    def ComputeEffAlgo_DiTauPNet(self, par):
        n_min_tau = 2

        # Get events that pass denominator
        events = self.get_events()

        # Selection at L1, reco and Gen Level for DiTauPNet HLT (without PNET_WP req)
        L1Tau_L2NN_mask = self.L1Tau_L2NN_selection_DiTau(events)
        L1Tau_IsoTau34er2p1_mask = self.L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_mask
        L1Tau_Tau70er2p1_mask = self.L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_mask
        GenTau_mask = self.GenTau_selection_DiTau(events)
        Jet_mask = self.Jet_selection_DiTau(events, par, apply_PNET_WP = False)

        evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_tau) | (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_tau)) & (ak.sum(GenTau_mask, axis=-1) >= n_min_tau) & (ak.sum(Jet_mask, axis=-1) >= n_min_tau) 
        events = events[evt_mask]
        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        GenTau_mask = GenTau_mask[evt_mask]
        Jet_mask = Jet_mask[evt_mask]

        L1Taus = self.get_L1Taus(events)
        GenTaus = self.get_GenTaus(events)
        Jets = self.get_Jets(events)  

        L1Taus_Sel = self.get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = n_min_tau)
        GenTaus_Sel = GenTaus[GenTau_mask]
        Jets_Sel = Jets[Jet_mask]

        # matching
        matching_GenTaus_mask = self.matching_Gentaus(L1Taus_Sel, Jets_Sel, GenTaus_Sel)

        # select events where at least n Tau that match Gen, L1, Jets exist
        evt_mask = ak.sum(matching_GenTaus_mask, axis=-1) >= n_min_tau
        events = events[evt_mask]

        N_den = len(events)
        print(f"Number of events with all HLT requirments excepted PNET_WP (N_den): {N_den}")

        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        GenTau_mask = GenTau_mask[evt_mask]
        Jet_mask = self.Jet_selection_DiTau(events, par, apply_PNET_WP = True)

        evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_tau) | (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_tau)) & (ak.sum(GenTau_mask, axis=-1) >= n_min_tau) & (ak.sum(Jet_mask, axis=-1) >= n_min_tau)
        events = events[evt_mask]

        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        GenTau_mask = GenTau_mask[evt_mask]
        Jet_mask = Jet_mask[evt_mask]

        # Selection at L1, reco and Gen Level for DiTauPNet HLT with PNET_WP req
        L1Taus = self.get_L1Taus(events)
        GenTaus = self.get_GenTaus(events)
        Jets = self.get_Jets(events)  

        L1Taus_Sel = self.get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = n_min_tau)
        GenTaus_Sel = GenTaus[GenTau_mask]
        Jets_Sel = Jets[Jet_mask]

        # matching
        matching_GenTaus_mask = self.matching_Gentaus(L1Taus_Sel, Jets_Sel, GenTaus_Sel)

        # select events where at least n Tau that match Gen, L1, Jets exist
        evt_mask = ak.sum(matching_GenTaus_mask, axis=-1) >= n_min_tau
        events = events[evt_mask]
        N_num = len(events)
        print(f"Number of events with all HLT requirments (N_num): {N_num}")
        print('')

        return N_den, N_num
    
    def ComputeEffAlgo_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(self):
        n_min_tau = 2

        # Get events that pass denominator
        events = self.get_events()

        # Selection at L1, reco and Gen Level for DiTauPNet HLT (without PNET_WP req)
        L1Tau_L2NN_mask = self.L1Tau_L2NN_selection_DiTau(events)
        L1Tau_IsoTau34er2p1_mask = self.L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_mask
        L1Tau_Tau70er2p1_mask = self.L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_mask
        GenTau_mask = self.GenTau_selection_DiTau(events)
        Tau_mask = self.Tau_selection_DiTau(events, apply_DeepTau_WP = False)

        evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_tau) | (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_tau)) & (ak.sum(GenTau_mask, axis=-1) >= n_min_tau) & (ak.sum(Tau_mask, axis=-1) >= n_min_tau) 
        events = events[evt_mask]
        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        GenTau_mask = GenTau_mask[evt_mask]
        Tau_mask = Tau_mask[evt_mask]

        L1Taus = self.get_L1Taus(events)
        GenTaus = self.get_GenTaus(events)
        Taus = self.get_Taus(events)  

        L1Taus_Sel = self.get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = n_min_tau)
        GenTaus_Sel = GenTaus[GenTau_mask]
        Taus_Sel = Taus[Tau_mask]

        # matching
        matching_GenTaus_mask = self.matching_Gentaus(L1Taus_Sel, Taus_Sel, GenTaus_Sel)

        # select events where at least n Tau that match Gen, L1, Taus exist
        evt_mask = ak.sum(matching_GenTaus_mask, axis=-1) >= n_min_tau
        events = events[evt_mask]

        N_den = len(events)
        print(f"Number of events with all HLT requirments excepted DeepTau_WP (N_den): {N_den}")

        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        GenTau_mask = GenTau_mask[evt_mask]
        Tau_mask = self.Tau_selection_DiTau(events, apply_DeepTau_WP = True)

        evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= n_min_tau) | (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= n_min_tau)) & (ak.sum(GenTau_mask, axis=-1) >= n_min_tau) & (ak.sum(Tau_mask, axis=-1) >= n_min_tau)
        events = events[evt_mask]

        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        GenTau_mask = GenTau_mask[evt_mask]
        Tau_mask = Tau_mask[evt_mask]

        # Selection at L1, reco and Gen Level for DiTauPNet HLT with PNET_WP req
        L1Taus = self.get_L1Taus(events)
        GenTaus = self.get_GenTaus(events)
        Taus = self.get_Taus(events)  

        L1Taus_Sel = self.get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = n_min_tau)
        GenTaus_Sel = GenTaus[GenTau_mask]
        Taus_Sel = Taus[Tau_mask]

        # matching
        matching_GenTaus_mask = self.matching_Gentaus(L1Taus_Sel, Taus_Sel, GenTaus_Sel)

        # select events where at least n Tau that match Gen, L1, Taus exist
        evt_mask = ak.sum(matching_GenTaus_mask, axis=-1) >= n_min_tau
        events = events[evt_mask]
        N_num = len(events)
        print(f"Number of events with all HLT requirments (N_num): {N_num}")
        print('')

        return N_den, N_num
# ------------------------------ functions for ComputeRate ---------------------------------------------------------------

    def get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(self):
        print(f'Computing rate for HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1:')
        #load all events in the file that belong to the run and lumiSections_range, and save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {len(events)}")

        # Selection of L1 objects and reco Tau objects for HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 path
        L1Tau_L2NN_mask = self.L1Tau_L2NN_selection_DiTau(events)
        L1Tau_IsoTau34er2p1_mask = self.L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_mask
        L1Tau_Tau70er2p1_mask = self.L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_mask
        Tau_mask = self.Tau_selection_DiTau(events)

        # at least 2 L1tau/ recoTau should pass
        evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= 2) | (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= 2)) & (ak.sum(Tau_mask, axis=-1) >= 2)
        events = events[evt_mask]
        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        Tau_mask = Tau_mask[evt_mask]
        print(f"Number of events with at least 2 Taus (L1 Tau and reco Tau) with HLT requirments: {len(events)}")

        # Get L1 and Tau to apply HLT condition
        L1Taus = self.get_L1Taus(events)
        Taus = self.get_Taus(events)

        L1Taus_Sel = self.get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = 2)
        Taus_Sel = Taus[Tau_mask]

        # do the matching between the 2 objects
        matching_taus_L1Taus_mask = self.matching_L1Taus_obj(L1Taus_Sel, Taus_Sel, dR_matching_min = 0.5)

        evt_mask = (ak.sum(matching_taus_L1Taus_mask, axis=-1) >= 2)  # at least 2 Taus should match L1Tau
        events = events[evt_mask]
        print(f"Number of events after matching: {len(events)}")
        print('')
        N_num = len(events)

        return N_den, N_num
    
    def get_Nnum_Nden_DiTauPNet(self, par):
        print(f'Computing Rate for DiTau path with param: {par}')
        #load all events in the file that belong to the run and lumiSections_range, and save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in run and lumiSections_range (N_den): {len(events)}")

        # Selection of L1 object and reco Jet objects for DiTauPNet path
        L1Tau_L2NN_mask = self.L1Tau_L2NN_selection_DiTau(events)
        L1Tau_IsoTau34er2p1_mask = self.L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_mask
        L1Tau_Tau70er2p1_mask = self.L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_mask
        Jet_mask = self.Jet_selection_DiTau(events, par)

        # at least 2 L1tau/ recoJet should pass
        evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1_mask, axis=-1) >= 2) | (ak.sum(L1Tau_Tau70er2p1_mask, axis=-1) >= 2)) & (ak.sum(Jet_mask, axis=-1) >= 2)
        events = events[evt_mask]
        L1Tau_IsoTau34er2p1_mask = L1Tau_IsoTau34er2p1_mask[evt_mask]
        L1Tau_Tau70er2p1_mask = L1Tau_Tau70er2p1_mask[evt_mask]
        Jet_mask = Jet_mask[evt_mask]
        print(f"Number of events with at least 2 Taus (L1Tau and Jet) with HLT requirments: {len(events)}")

        # Get L1 and Jets to apply HLT condition
        L1Taus = self.get_L1Taus(events)
        Jets = self.get_Jets(events)

        L1Taus_Sel = self.get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1_mask, L1Tau_Tau70er2p1_mask, n_min_taus = 2)
        Jets_Sel = Jets[Jet_mask]

        # do the matching between the 2 objects
        matching_jets_L1Taus_mask = self.matching_L1Taus_obj(L1Taus_Sel, Jets_Sel, dR_matching_min = 0.5)

        evt_mask = (ak.sum(matching_jets_L1Taus_mask, axis=-1) >= 2) # at least 2 Jets should match L1Tau
        events = events[evt_mask]
        print(f"Number of events after matching (N_num): {len(events)}")
        print('')
        N_num = len(events)

        return N_den, N_num
    
