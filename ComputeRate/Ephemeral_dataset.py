import uproot
import collections
import six
import numpy as np

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
    
    def evt_run_selection(self, run = 362439, lumiSections_range = [35,249]):
        events = self.get_events()
        print(f"Number of events: {len(events)}")
        events = events[events['run'] == run]
        print(f"Number of events belonging to run {run}: {len(events)}")
        events = events[(events["luminosityBlock"] >= lumiSections_range[0]) & (events["luminosityBlock"] <= lumiSections_range[1])]
        print(f"Number of event in LumiSections range {lumiSections_range}: {len(events)}")
        return events
    
    def compute_rate(self, run = 362439, lumiSections_range = [35,249], HLTname = "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1"):
        events = self.evt_run_selection(run = run, lumiSections_range = lumiSections_range)
        N_den = len(events)
        events_HLT = events[events[HLTname]]
        print(f"Number of event that pass {HLTname}: {len(events_HLT)}")
        N_num = len(events_HLT)
        return N_den, N_num