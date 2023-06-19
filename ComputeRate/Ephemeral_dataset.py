import uproot
import collections
import six
import numpy as np
import awkward as ak

def iterable(arg):
    return (
        isinstance(arg, collections.abc.Iterable)
        and not isinstance(arg, six.string_types)
    )

class Ephemeral:
    def __init__(self, fileName):
        self.fileName = fileName

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
        events = uproot.lazy(tree_path)
        return events
    
    def get_events_run_lumi(self, run, lumiSections_range):
        events = self.get_events()
        print(f"Number of events: {len(events)}")
        events = events[events['run'] == run]
        print(f"Number of events belonging to run {run}: {len(events)}")
        events = events[(events["luminosityBlock"] >= lumiSections_range[0]) & (events["luminosityBlock"] <= lumiSections_range[1])]
        print(f"Number of event in LumiSections range {lumiSections_range}: {len(events)}")
        print(f"   - events passing L1_DoubleIsoTau34er2p1 flag: {len(events[events.L1_DoubleIsoTau34er2p1])}")
        print(f"   - events passing HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 flag: {len(events[events.HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1])}")
        return events
    
    def L1DoubleIsoTau34er2p1_selection(self, events):
        # Apply L1_DoubleIsoTau34er2p1 filter 
        l1_mask = (events.L1Tau_hwPt >= 0x44) & (events.L1Tau_hwEta <= 0x30) & (events.L1Tau_eta >= -2.131) & (events.L1Tau_hwIso > 0 )
        ev_mask = ak.sum(l1_mask, axis=-1) >= 2 # at least 2 taus with pt eta iso
        events = events[ev_mask]
        print(f"Number of events with at least 2 isolated L1 taus (L1Tau_hwIso > 0) with pt >= 34 and |eta| <= 2.131: {len(events)}")
        return events

    def L1tau_selection(self, events):
        # Apply L2 filter to Di-Tau path
        l2_mask = (events.L1Tau_l2Tag > 0.386) | (events.L1Tau_pt >= 250)
        ev_mask = ak.sum(l2_mask, axis=-1) >= 2  # at least 2 taus with l2Tag >= 0.386 or L1Tau_pt >= 250
        events = events[ev_mask]
        print(f"Number of events with at least 2 L1 taus with l2Tag >= 0.386 or pt >= 250: {len(events)}")
        return events
    
    def compute_WP(self, tau_pt):
        # should return WP with same shape as tau_pt
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

    def reco_tau_selection(self, events):
        tau_mask = (events.Tau_pt >= 35) & (np.abs(events.Tau_eta) <= 2.1) & (events.Tau_deepTauVSjet >= self.compute_WP(events.Tau_pt))
        ev_mask = ak.sum(tau_mask, axis=-1) >= 2 # at least 2 taus with pt eta deepTauVSjet
        events = events[ev_mask]
        print(f"Number of events with at least 2 reco taus with pt >= 35, |eta| <= 2.1 and deepTauVSjet >= (anna's formula): {len(events)}")
        return events

    def get_taus(self, events):
        taus_dict = {"pt": events["Tau_pt"], "eta": events["Tau_eta"], "phi": events["Tau_phi"]}
        taus = ak.zip(taus_dict)
        index = ak.argsort(taus.pt, ascending=False)
        taus = taus[index]
        return taus

    def get_L1taus(self, events):
        L1taus_dict = {"pt": events["L1Tau_pt"], "eta": events["L1Tau_eta"], "phi": events["L1Tau_phi"]}
        L1taus = ak.zip(L1taus_dict)
        index = ak.argsort(L1taus.pt, ascending=False)
        L1taus = L1taus[index]
        return L1taus

    def delta_r2(self, v1, v2):
        dphi = (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi
        deta = v1.eta - v2.eta
        dr2 = dphi ** 2 + deta ** 2
        return dr2

    def delta_r(self, v1, v2):
        return np.sqrt(self.delta_r2(v1, v2))

    def matching_L1taus_taus(self, events, taus, L1taus):
        dR_matching = 0.5
        tau_inpair, L1_inpair = ak.unzip(ak.cartesian([taus, L1taus], nested=True))
        dR = self.delta_r(tau_inpair, L1_inpair)
        # check events where at least 2 reco taus match 2 L1 taus
        mask = (dR < dR_matching)
        ev_mask = ak.sum(ak.sum(mask, axis=-1) >= 1, axis=-1) >= 2 # careful, here L1 could match twice
        events = events[ev_mask]
        print(f"Number of events with matching reco Tau and L1Tau: {len(events)}")
        return events

    def get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(self, run, lumiSections_range):
        events = self.get_events_run_lumi(run, lumiSections_range)
        N_den = len(events)
        events = self.L1DoubleIsoTau34er2p1_selection(events)
        events = self.L1tau_selection(events)
        events = self.reco_tau_selection(events)
        taus = self.get_taus(events)
        L1taus = self.get_L1taus(events)
        taus = taus[taus.pt >= 35] # we should have at least 2 taus satisfying this condition
        L1taus = L1taus[L1taus.pt >= 34] # we should have at least 2 taus satisfying this condition
        if ak.sum((ak.num(taus, axis=-1) < 2) | (ak.num(L1taus, axis=-1) < 2)) != 0:
            raise('Error in selection')
        events = self.matching_L1taus_taus(events, taus, L1taus)
        N_num = len(events)
        return N_den, N_num

    def get_Nnum_Nden_L1(self, run, lumiSections_range):
        events = self.get_events_run_lumi(run, lumiSections_range)
        N_den = len(events)
        events = self.L1DoubleIsoTau34er2p1_selection(events)
        N_num = len(events)
        return N_den, N_num
    
    def get_Nnum_Nden_HLTflag(self, run, lumiSections_range, HLTname):
        events = self.get_events_run_lumi(run, lumiSections_range)
        N_den = len(events)
        events_HLT = events[events[HLTname]]
        print(f"Number of event that pass {HLTname}: {len(events_HLT)}")
        N_num = len(events_HLT)
        return N_den, N_num