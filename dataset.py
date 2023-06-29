import uproot
import collections
import six
import numpy as np
import awkward as ak
from helpers import delta_r

def iterable(arg):
    return (
        isinstance(arg, collections.abc.Iterable)
        and not isinstance(arg, six.string_types)
    )

class Dataset:
    def __init__(self, fileName):
        self.fileName = fileName

    @staticmethod
    def compute_decay_mode(nChargedHad, nNeutralHad):
        return (nChargedHad - 1) * 5 + nNeutralHad

    @staticmethod
    def compute_PNet_charge_prob(probTauP, probTauM):
        return np.abs(np.ones(len(probTauP))*0.5 - probTauP/(probTauP + probTauM))

    @staticmethod
    def compute_DeepTau_WP(tau_pt):
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
    
    def __define_tree_expression(self):
        treeName = 'Events'
        if iterable(self.fileName):
            tree_path = []
            for file in self.fileName:
                tree_path.append(file + ":" + treeName)
        else:
            tree_path = self.fileName + ":" + treeName
        return tree_path

    def get_events(self):
        tree_path = self.__define_tree_expression()
        events = uproot.dask(tree_path, library='ak')
        return events

    def get_events_run_lumi(self, run, lumiSections_range, debug_mode = True):
        events = self.get_events()
        if debug_mode:
            print(f"Number of events: {len(events)}")
        events = events[events['run'] == run]
        if debug_mode:
            print(f"Number of events belonging to run {run}: {len(events)}")
        events = events[(events["luminosityBlock"] >= lumiSections_range[0]) & (events["luminosityBlock"] <= lumiSections_range[1])]
        if debug_mode:
            print(f"Number of event in LumiSections range {lumiSections_range}: {len(events)}")
            print(f"   - events passing L1_DoubleIsoTau34er2p1 flag: {len(events[events['L1_DoubleIsoTau34er2p1'].compute()])}")
            print(f"   - events passing HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 flag: {len(events[events['HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1'].compute()])}")

        return events

    def get_GenLepton(self, events):
        from ComputeEfficiency.GenLeptonCode.helpers import get_GenLepton_rdf
        # Add GenLepton information to RdataFrame
        rdf = ak.to_rdataframe({'GenPart_pt': events['GenPart_pt'].compute(),
                                'GenPart_eta': events['GenPart_eta'].compute(),
                                'GenPart_phi': events['GenPart_phi'].compute(),
                                'GenPart_mass': events['GenPart_mass'].compute(),
                                'GenPart_genPartIdxMother': events['GenPart_genPartIdxMother'].compute(),
                                'GenPart_pdgId': events['GenPart_pdgId'].compute(),
                                'GenPart_statusFlags': events['GenPart_statusFlags'].compute()})

        rdf = get_GenLepton_rdf(rdf)
        GenLepton = {}
        GenLepton['pt'] = ak.from_rdataframe(rdf, 'GenLepton_pt')
        GenLepton['eta'] = ak.from_rdataframe(rdf, 'GenLepton_eta')
        GenLepton['phi'] = ak.from_rdataframe(rdf, 'GenLepton_phi')
        GenLepton['kind'] = ak.from_rdataframe(rdf, 'GenLepton_kind')
        #GenLepton['nChargedHad'] = ak.from_rdataframe(rdf, 'GenLepton_nChargedHad')
        #GenLepton['nNeutralHad'] = ak.from_rdataframe(rdf, 'GenLepton_nNeutralHad')
        #GenLepton['DecayMode'] = self.compute_decay_mode(GenLepton['nChargedHad'], GenLepton['nNeutralHad'])

        return GenLepton

    def Denominator_Selection(self, GenLepton):
        # For diTau HLT: 2 hadronic Gen Tau that match vis. pt and eta requirements
        mask = (GenLepton['pt'] >= 20) & (np.abs(GenLepton['eta']) <= 2.1) & (GenLepton['kind'] == 5)
        ev_mask = ak.sum(mask, axis=-1) >= 2  # at least 2 Gen taus should pass this requirements
        print(f"Number of events with at least 2 Gen Tau with vis. pt >= 20, |eta|<2.1 and kind=5 (TauDecayedToHadrons): {ak.sum(ev_mask)}")
        return ev_mask

    def L1Tau_selection(self, events):
        # Apply L1_DoubleIsoTau34er2p1 filter 
        L1Tau_mask = (events['L1Tau_hwPt'].compute() >= 0x44) & (events['L1Tau_hwEta'].compute() <= 0x30) & (events['L1Tau_hwEta'].compute() >= -49) & (events['L1Tau_hwIso'].compute() > 0 ) & ((events['L1Tau_l2Tag'].compute() > 0.386) | (events['L1Tau_pt'].compute() >= 250))
        return L1Tau_mask
    
    def Tau_selection(self, events):
        # Apply reco Tau selection
        Tau_mask = (events['Tau_pt'].compute() >= 35) & (np.abs(events['Tau_eta'].compute()) <= 2.1) & (events['Tau_deepTauVSjet'].compute() >= self.compute_DeepTau_WP(events['Tau_pt'].compute()))
        return Tau_mask
    
    def GenTau_selection(self, events):
        # Minimal requirements so that event pass 
        GenTau_mask = (events['GenLepton_pt'].compute() >= 20) & (np.abs(events['GenLepton_eta'].compute()) <= 2.1) & (events['GenLepton_kind'].compute() == 5)
        return GenTau_mask

    def Jet_selection(self, events, treshold):
        Jets_mask = (events['Jet_pt'].compute()*events['Jet_PNet_ptcorr'].compute() >= 30) & (np.abs(events['Jet_eta'].compute()) <= 2.5) & ((events['Jet_PNet_probtauhp'].compute() >= treshold) | (events['Jet_PNet_probtauhm'].compute() >= treshold))
        return Jets_mask

    def get_L1Taus(self, events):
        L1taus_dict = {"pt": events["L1Tau_pt"].compute(), "eta": events["L1Tau_eta"].compute(), "phi": events["L1Tau_phi"].compute(), "Iso": events["L1Tau_hwIso"].compute()}
        L1Taus = ak.zip(L1taus_dict)
        return L1Taus

    def get_Taus(self, events):
        taus_dict = {"pt": events["Tau_pt"].compute(), "eta": events["Tau_eta"].compute(), "phi": events["Tau_phi"].compute(), "deepTauVSjet": events["Tau_deepTauVSjet"].compute()}
        Taus = ak.zip(taus_dict)
        return Taus

    def get_GenTaus(self, events):
        gentaus_dict = {"pt": events["GenLepton_pt"].compute(), "eta": events["GenLepton_eta"].compute(), "phi": events["GenLepton_phi"].compute()}
        GenTaus = ak.zip(gentaus_dict)
        return GenTaus

    def get_Jets(self, events):
        jets_dict = {"pt": events["Jet_PNet_ptcorr"].compute()*events["Jet_pt"].compute(), "eta": events["Jet_eta"].compute(), "phi": events["Jet_phi"].compute(), "probtauhm": events['Jet_PNet_probtauhm'].compute(), "probtauhp": events['Jet_PNet_probtauhp'].compute()}
        Jets = ak.zip(jets_dict)
        return Jets

    def matching_taus(self, L1Taus, GenTaus, Taus, dR_matching_min = 0.5):
        
        taus_inpair, l1taus_inpair = ak.unzip(ak.cartesian([Taus, L1Taus], nested=True))
        dR_taus_l1taus = delta_r(taus_inpair, l1taus_inpair)
        mask_taus_l1taus = (dR_taus_l1taus < dR_matching_min) 

        taus_inpair, gentaus_inpair = ak.unzip(ak.cartesian([Taus, GenTaus], nested=True))
        dR_taus_gentaus = delta_r(taus_inpair, gentaus_inpair)
        mask_taus_gentaus = (dR_taus_gentaus < dR_matching_min)

        matching_taus_mask = ak.any(mask_taus_l1taus, axis=-1) & ak.any(mask_taus_gentaus, axis=-1)  # tau should match l1 and gentau

        return matching_taus_mask

    def matching_Taus_obj(self, ObjToMatch, Taus, dR_matching_min = 0.5):
        taus_inpair, obj_inpair = ak.unzip(ak.cartesian([Taus, ObjToMatch], nested=True))
        dR_taus_obj = delta_r(taus_inpair, obj_inpair)
        mask_taus_obj = (dR_taus_obj < dR_matching_min)   
        matching_taus_mask = ak.any(mask_taus_obj, axis=-1)

        return matching_taus_mask
    
    def save_info(self, Tau_Den, Tau_Num, out_file):
        # saving infos
        lst_Den = {}
        lst_Den['Tau_pt'] = Tau_Den.pt
        lst_Den['Tau_eta'] = Tau_Den.eta
        lst_Den['Tau_phi'] = Tau_Den.phi

        lst_Num = {}
        lst_Num['Tau_pt'] = Tau_Num.pt
        lst_Num['Tau_eta'] = Tau_Num.eta
        lst_Num['Tau_phi'] = Tau_Num.phi

        with uproot.create(out_file, compression=uproot.ZLIB(4)) as file:
            file["TausDen"] = lst_Den
            file["TausNum"] = lst_Num
        return

# ------------------------------ functions for ComputeEfficiency ---------------------------------------------------------------
    def save_Event_Nden(self, tmp_file):
        ''' 
        Save only needed informations (for numerator cuts) of events passing denominator cuts (Denominator_Selection)
        '''
        events = self.get_events()
        print(f"Number of events in the file: {len(events)}")
        GenLepton = self.get_GenLepton(events)
        ev_mask = self.Denominator_Selection(GenLepton)

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
    
    def produceRoot_DiTauPNet(self, out_file, treshold):
        n_min_tau = 1

        # Get events that pass denominator
        events = self.get_events()
        # N events in denominator passing Denominator_Selection
        print(f"Number of events in the denominator: {len(events)}")

        # Selection at L1, reco and Gen Level for DiTauPNet HLT
        L1Tau_mask = self.L1Tau_selection(events)
        Jet_mask = self.Jet_selection(events, treshold)
        GenTau_mask = self.GenTau_selection(events)

        # To compute efficiency, we save in denominator GenTau which pass minimal selection (at least 2 /events)
        Tau_Den = self.get_GenTaus(events)
        Tau_Den = Tau_Den[GenTau_mask]

        evt_mask = (ak.sum(L1Tau_mask, axis=-1) >= n_min_tau) & (ak.sum(GenTau_mask, axis=-1) >= n_min_tau) & (ak.sum(Jet_mask, axis=-1) >= n_min_tau) 
        events = events[evt_mask]
        print(f"Number of events with at least {n_min_tau} Tau (L1Tau, GenTau, and Jet) with HLT requirments: {len(events)}")

        L1Taus = self.get_L1Taus(events)
        GenTaus = self.get_GenTaus(events)
        Jets = self.get_Jets(events)  

        GenTaus = GenTaus[GenTau_mask[evt_mask]]
        #L1Taus = L1Taus[L1Tau_mask[evt_mask]]
        #Jets = Jets[Jet_mask[evt_mask]]
        # ------------------
        # apply all condition except L2NN and Iso on L1Tau
        L1Taus = L1Taus[(L1Taus.pt >= 34) & (np.abs(L1Taus.eta) <= 2.131)]
        # apply all condition except PNet Treshold Jet
        Jets = Jets[(Jets.pt >= 30) & (np.abs(Jets.eta) <= 2.5)]
        # ------------------

        # matching
        matching_GenTaus_mask = self.matching_taus(L1Taus, Jets, GenTaus)

        # select events where at least n Tau that match Gen, L1, Jets exist
        evt_mask = ak.sum(matching_GenTaus_mask, axis=-1) >= n_min_tau
        
        events = events[evt_mask]
        Tau_Num = (GenTaus[matching_GenTaus_mask])[evt_mask]

        print(f"Number of events after matching: {len(events)}")

        self.save_info(Tau_Den, Tau_Num, out_file)

        return

    def produceRoot_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(self, out_file):
        n_min_tau = 1

        # Get events that pass denominator
        events = self.get_events()
        # N events in denominator passing Denominator_Selection
        print(f"Number of events in the denominator: {len(events)}")

        # Selection at L1, reco and Gen Level for HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1
        L1Tau_mask = self.L1Tau_selection(events)
        Tau_mask = self.Tau_selection(events)
        GenTau_mask = self.GenTau_selection(events)

        # To compute efficiency, we save in denominator GenTau which pass minimal selection (at least 2 /events)
        Tau_Den = self.get_GenTaus(events)
        Tau_Den = Tau_Den[GenTau_mask]

        evt_mask = (ak.sum(L1Tau_mask, axis=-1) >= n_min_tau) & (ak.sum(GenTau_mask, axis=-1) >= n_min_tau)& (ak.sum(Tau_mask, axis=-1) >= n_min_tau)
        events = events[evt_mask]
        print(f"Number of events with at least {n_min_tau} Tau (L1Tau, GenTau and reco Tau) with HLT requirments: {len(events)}")

        L1Taus = self.get_L1Taus(events)
        GenTaus = self.get_GenTaus(events)
        Taus = self.get_Taus(events)

        GenTaus = GenTaus[GenTau_mask[evt_mask]]
        #L1Taus = L1Taus[L1Tau_mask[evt_mask]]
        #Taus = Taus[Tau_mask[evt_mask]]
        # ------------------
        # apply all condition except L2NN and Iso on L1Tau
        L1Taus = L1Taus[(L1Taus.pt >= 34) & (np.abs(L1Taus.eta) <= 2.131)]
        # apply all condition except DeepTau on Tau
        Taus = Taus[(Taus.pt >= 35) & (np.abs(Taus.eta) <= 2.1)]
        # ------------------

        # matching
        matching_GenTaus_mask = self.matching_taus(L1Taus, Taus, GenTaus)

        # select events where at least n Tau that match Gen, L1, Jets exist
        evt_mask = ak.sum(matching_GenTaus_mask, axis=-1) >= n_min_tau
        
        events = events[evt_mask]
        Tau_Num = (GenTaus[matching_GenTaus_mask])[evt_mask]

        print(f"Number of events after matching: {len(events)}")

        self.save_info(Tau_Den, Tau_Num, out_file)

        return

# ------------------------------ functions for ComputeRate ---------------------------------------------------------------
    def get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(self, run, lumiSections_range):
        #load all events in the file that belong to the run and lumiSections_range, and save the number of events in Denominator
        events = self.get_events_run_lumi(run, lumiSections_range)
        N_den = len(events)

        # Selection of L1 object and reco Tau objects for HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 path
        L1Tau_mask = self.L1Tau_selection(events)
        Tau_mask = self.Tau_selection(events)
        evt_mask = (ak.sum(L1Tau_mask, axis=-1) >= 2) & (ak.sum(Tau_mask, axis=-1) >= 2)
        events = events[evt_mask]
        print(f"Number of events with at least 2 Taus (L1 Tau and reco Tau) with HLT requirments: {len(events)}")

        # Get L1 and Tau to apply HLT condition
        L1Taus = self.get_L1Taus(events)
        Taus = self.get_Taus(events) 
        #Taus = Taus[Tau_mask[evt_mask]]
        #L1Taus = L1Taus[L1Tau_mask[evt_mask]]
        # ------------------
        # apply all condition except L2NN and Iso on L1Tau
        L1Taus = L1Taus[(L1Taus.pt >= 34) & (np.abs(L1Taus.eta) <= 2.131)]
        # apply all condition except DeepTau on Tau
        Taus = Taus[(Taus.pt >= 35) & (np.abs(Taus.eta) <= 2.1)]
        # ------------------
        # do the matching between the 2 objects
        matching_taus_mask = self.matching_Taus_obj(L1Taus, Taus, dR_matching_min = 0.5)

        '''
        index = 17
        print(f'Taus.pt : {Taus.pt[index]}')
        print(f'L1Taus.pt : {L1Taus.pt[index]}')
        print(f'L1Taus.Iso : {L1Taus.Iso[index]}')
        print(f'L1Taus.L2NN : {(events["L1Tau_l2Tag"].compute()[(L1Taus.pt >= 34) & (np.abs(L1Taus.eta) <= 2.131)])[index]}')
        taus_inpair, obj_inpair = ak.unzip(ak.cartesian([Taus, L1Taus], nested=True))
        dR_taus_obj = delta_r(taus_inpair, obj_inpair)
        print(f'dR :')
        print(dR_taus_obj[index])
        print(f'matching_taus_mask : {matching_taus_mask[index]}')
        '''

        evt_mask = (ak.sum(matching_taus_mask, axis=-1) >= 2) # at least 2 Taus should match L1Tau
        events = events[evt_mask]
        print(f"Number of events after matching: {len(events)}")
        print('')
        N_num = len(events)
        return N_den, N_num
    
    def get_Nnum_Nden_DiTauPNet(self, run, lumiSections_range, treshold):
        #load all events in the file that belong to the run and lumiSections_range, and save the number of events in Denominator
        events = self.get_events_run_lumi(run, lumiSections_range)
        N_den = len(events)

        # Selection of L1 object and reco Jet objects for DiTauPNet path
        L1Tau_mask = self.L1Tau_selection(events)
        Jet_mask = self.Jet_selection(events, treshold)
        evt_mask = (ak.sum(L1Tau_mask, axis=-1) >= 2) & (ak.sum(Jet_mask, axis=-1) >= 2)
        events = events[evt_mask]
        print(f"Number of events with at least 2 Taus (L1 Tau and reco Jet) with HLT requirments: {len(events)}")

        # Get L1 and Jets to apply HLT condition
        L1Taus = self.get_L1Taus(events)
        Jets = self.get_Jets(events) 
        #Jets = Jets[Jet_mask[evt_mask]]
        #L1Taus = L1Taus[L1Tau_mask[evt_mask]]
        # ------------------
        # apply all condition except L2NN and Iso on L1Tau
        L1Taus = L1Taus[(L1Taus.pt >= 34) & (np.abs(L1Taus.eta) <= 2.131)]
        # apply all condition except PNet Treshold Jet
        Jets = Jets[(Jets.pt >= 30) & (np.abs(Jets.eta) <= 2.5)]
        # ------------------

        # do the matching between the 2 objects
        matching_jets_mask = self.matching_Taus_obj(L1Taus, Jets, dR_matching_min = 0.5)
        evt_mask = (ak.sum(matching_jets_mask, axis=-1) >= 2) # at least 2 Jets should match L1Tau
        events = events[evt_mask]
        print(f"Number of events after matching: {len(events)}")
        print('')
        N_num = len(events)
        return N_den, N_num