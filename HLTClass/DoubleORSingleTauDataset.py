import awkward as ak
import numpy as np
from HLTClass.dataset import Dataset
from HLTClass.dataset import get_L1Taus, get_Taus, get_Jets, get_GenTaus, hGenTau_selection, matching_Gentaus, matching_L1Taus_obj
from helpers import load_cfg_file

from HLTClass.DiTauDataset import evt_sel_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1, evt_sel_DiTau, Denominator_Selection_DiTau, L1Tau_IsoTau34er2p1_selection,L1Tau_Tau70er2p1_selection, L1Tau_L2NN_selection_DiTau, Jet_selection_DiTau, get_selL1Taus
from HLTClass.SingleTauDataset import evt_sel_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3, evt_sel_SingleTau, Denominator_Selection_SingleTau, Jet_selection_SingleTau,L1Tau_Tau120er2p1_selection, L1Tau_L2NN_selection_SingleTau

class DoubleORSingleTauDataset(Dataset):
    config = load_cfg_file()
    par_frozen_SingleTau = [float(config['OPT']['PNet_SingleTau_t1']), float(config['OPT']['PNet_SingleTau_t2']), float(config['OPT']['PNet_SingleTau_t3'])]

    def __init__(self, fileName):
        Dataset.__init__(self, fileName)
        
# ------------------------------ functions for ComputeRate ---------------------------------------------------------------

    def get_Nnum_Nden_HLT_DoubleORSingleDeepTau(self):
        print(f'Computing rate for HLT_DoubleORSingleDeepTau:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        SingleTau_evt_mask, SingleTau_matchingTaus_mask = evt_sel_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(events, is_gen = False)
        DiTau_evt_mask, DiTau_matchingTaus_mask = evt_sel_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(events, n_min=2, is_gen = False)

        print(np.sum(SingleTau_evt_mask))
        print(np.sum(DiTau_evt_mask))
        evt_mask = SingleTau_evt_mask | DiTau_evt_mask

        N_num = len(events[evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num


    def get_Nnum_Nden_DoubleORSinglePNet(self, par):
        print(f'Computing Rate for DoubleORSingle path with param: {par}')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        SingleTau_evt_mask, SingleTau_matchingTaus_mask = evt_sel_SingleTau(events, self.par_frozen_SingleTau, is_gen = False)
        DiTau_evt_mask, DiTau_matchingTaus_mask = evt_sel_DiTau(events, par, n_min=2, is_gen = False)

        evt_mask = SingleTau_evt_mask | DiTau_evt_mask

        N_num = len(events[evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num
    
# ------------------------------ functions for ComputeEfficiency ---------------------------------------------------------------

    def save_Event_Nden_eff_DoubleORSingleTau(self, tmp_file):
        #Save only needed informations for events passing minimal Gen requirements for diTau HLT (passing denominator selection for efficiency)
        events = self.get_events()
        print(f"Number of events in the file: {len(events)}")
        GenLepton = self.get_GenLepton(events)
        evt_mask_SingleTau = Denominator_Selection_SingleTau(GenLepton)
        evt_mask_DiTau = Denominator_Selection_DiTau(GenLepton)
        evt_mask = (evt_mask_SingleTau | evt_mask_DiTau)
        print(f"Number of events with exactly 1 hadronic Tau (kind=5) or events with at least 2 Gen Tau: {ak.sum(evt_mask)}")
        self.Save_Event_Nden_Eff(events, GenLepton, evt_mask, tmp_file)
        return

    def produceRoot_DoubleORSingleDeepTau(self, out_file):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]

        mask_den_selection = ak.num(Tau_Den['pt']) >=2
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")
        SingleTau_evt_mask, SingleTau_matchingGentaus_mask = evt_sel_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(events, is_gen = True)
        DiTau_evt_mask, DiTau_matchingTaus_mask = evt_sel_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(events, n_min = 2, is_gen = True)
        # Or between the 2 
        evt_mask = SingleTau_evt_mask | DiTau_evt_mask

        Tau_Num = (Tau_Den[SingleTau_matchingGentaus_mask|DiTau_matchingTaus_mask])[evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")
        events_Num= events[evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, out_file)

        return  

    def produceRoot_DoubleORSinglePNet(self, out_file, par):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]

        mask_den_selection = ak.num(Tau_Den['pt']) >=2
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        SingleTau_evt_mask, SingleTau_matchingGentaus_mask = evt_sel_SingleTau(events, self.par_frozen_SingleTau, is_gen = True)
        DiTau_evt_mask, DiTau_matchingTaus_mask = evt_sel_DiTau(events, par, n_min=2, is_gen = True)

        print(f'SingleTau_evt_mask: {np.sum(SingleTau_evt_mask)}')
        print(f'DiTau_evt_mask: {np.sum(DiTau_evt_mask)}')

        evt_mask = SingleTau_evt_mask | DiTau_evt_mask

        print(f'evt_mask: {np.sum(evt_mask)}')

        print(f'SingleTau_matchingGentaus_mask: {np.sum(SingleTau_matchingGentaus_mask[evt_mask])}')
        print(f'DiTau_matchingTaus_mask: {np.sum(DiTau_matchingTaus_mask[evt_mask])}')

        Tau_Num = (Tau_Den[SingleTau_matchingGentaus_mask|DiTau_matchingTaus_mask])[evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")
        events_Num= events[evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, out_file)
        return

    # ------------------------------ functions to Compute Efficiency for opt ---------------------------------------------------------------

    def ComputeEffAlgo_DoubleORSinglePNet(self, par):

        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        L1Taus = get_L1Taus(events)
        Jets = get_Jets(events)
        GenTaus = get_GenTaus(events)

        #Select GenTau (same for both)
        GenTau_mask = hGenTau_selection(events)
        GenTaus_Sel = GenTaus[GenTau_mask]

        # Selection of L1 objects and reco Tau objects + matching
        # For SingleTau
        L1Tau_Tau120er2p1L2NN_mask = L1Tau_Tau120er2p1_selection(events) & L1Tau_L2NN_selection_SingleTau(events)
        SingleTau_mask = Jet_selection_SingleTau(events, self.par_frozen_SingleTau, apply_PNET_WP = False)
        # at least 1 L1tau/ recoTau should pass
        SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        L1Taus_SingleTau = L1Taus[L1Tau_Tau120er2p1L2NN_mask]
        Jets_SingleTau = Jets[SingleTau_mask]
        matching_GenTaus_mask_SingleTau = matching_Gentaus(L1Taus_SingleTau, Jets_SingleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matching_GenTaus_mask_SingleTau, axis=-1) >= 1)  # at least 1 Taus should match L1Tau
        SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching

        # For DiTau
        L1Tau_IsoTau34er2p1L2NN_mask = L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
        L1Tau_Tau70er2p1L2NN_mask = L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
        DiTau_mask = Jet_selection_DiTau(events, par, apply_PNET_WP = False)
        # at least n_min L1tau/ recoJet should pass
        DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= 2) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= 2)) & (ak.sum(DiTau_mask, axis=-1) >= 2) & (ak.sum(GenTau_mask, axis=-1) >= 2)

        # matching
        L1Taus_DoubleTau = get_selL1Taus(L1Taus, L1Tau_IsoTau34er2p1L2NN_mask, L1Tau_Tau70er2p1L2NN_mask, n_min_taus = 2)
        Jets_DoubleTau = Jets[DiTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Jets_DoubleTau, GenTaus_Sel)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 2)
        DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching

        # Or between the 2 
        evt_mask = SingleTau_evt_mask | DiTau_evt_mask

        N_den = len(events[evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1 objects and reco Tau objects + matching
        # For SingleTau
        SingleTau_mask = Jet_selection_SingleTau(events, self.par_frozen_SingleTau, apply_PNET_WP = True)
        # at least 1 L1tau/ recoTau should pass
        SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        Jets_SingleTau = Jets[SingleTau_mask]
        matching_GenTaus_mask_SingleTau = matching_Gentaus(L1Taus_SingleTau, Jets_SingleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matching_GenTaus_mask_SingleTau, axis=-1) >= 1)  # at least 1 Taus should match L1Tau
        SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching

        # For DiTau
        DiTau_mask = Jet_selection_DiTau(events, par, apply_PNET_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= 2) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= 2)) & (ak.sum(DiTau_mask, axis=-1) >= 2) & (ak.sum(GenTau_mask, axis=-1) >= 2)

        # matching
        Jets_DoubleTau = Jets[DiTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Jets_DoubleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 2)
        DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching

        # Or between the 2 
        evt_mask = SingleTau_evt_mask | DiTau_evt_mask
        
        N_num = len(events[evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num
