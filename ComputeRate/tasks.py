#!/usr/bin/env python
import law
import os
from dataset import Dataset
from helpers import files_from_path, load_cfg_file
from law_customizations import Task, HTCondorWorkflow

class ProduceRateFiles(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce json files where number of events which pass denominator and numerator criteria are saved from each ephemerals files
    '''
    config = load_cfg_file()
    RefRun = int(config['RUNINFO']['ref_run'])
    LumiSectionsRange = [int(config['RUNINFO']['LumiSectionsRange_low']), int(config['RUNINFO']['LumiSectionsRange_up'])]
    number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
    HLT_name = config['HLT']['HLTname']
    EphemeralFolder = config['DATA']['SamplesPath']
    runArea = config['RUNINFO']['Area']
    PNet_treshold = config['OPT']['PNet_treshold']
    output_path = os.path.join(config['DATA']['result_rate'], f'result_{RefRun}')
    
    if PNet_treshold == 'None':
        PNetConfig = False
        output_path = os.path.join(output_path, HLT_name)
    else:
        PNetConfig = True
        PNetTreshold = float(PNet_treshold)
        output_path = os.path.join(output_path, f'PNetTresh_{PNetTreshold}')

    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(self.number_of_ephemeral_folder):
            branches[i] = os.path.join(self.output_path, f'folder_{str(i)}.json')
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        print(f"-----------------------------------------------Processing folder {self.branch}-----------------------------------------------")

        EphemeralFolderName = os.path.join(self.EphemeralFolder, f"EphemeralHLTPhysics{self.branch}_Run{self.runArea}/")
        event_counter = {}
        FileNameList = files_from_path(EphemeralFolderName)
        for FileName in FileNameList:
            # this tmp folder fuck up the code
            if FileName == os.path.join(self.EphemeralFolder, 'EphemeralHLTPhysics1_Run2023C', 'nano.tmp'):
                continue
            event_counter[FileName] = {}
            print(f"For {FileName.replace(self.EphemeralFolder, '')}:")
            eph_dataset = Dataset(FileName)

            if self.PNetConfig:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_DiTauPNet(run = self.RefRun, lumiSections_range = self.LumiSectionsRange, treshold = self.PNetTreshold)
            else:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(run = self.RefRun, lumiSections_range = self.LumiSectionsRange)

            event_counter[FileName]['N_den'] = N_den_i
            event_counter[FileName]['N_num'] = N_num_i

        self.output().dump(event_counter)
