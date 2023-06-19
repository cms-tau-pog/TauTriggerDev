#!/usr/bin/env python
import law
import os
from ComputeRate.Ephemeral_dataset import Ephemeral
from helpers import files_from_path, load_cfg_file_ComputeRate
from law_customizations import Task, HTCondorWorkflow

class ProduceRateFiles(Task, HTCondorWorkflow, law.LocalWorkflow):
    config = load_cfg_file_ComputeRate()
    RefRun = int(config['RUNINFO']['ref_run'])
    LumiSectionsRange_low = int(config['RUNINFO']['LumiSectionsRange_low'])
    LumiSectionsRange_up = int(config['RUNINFO']['LumiSectionsRange_up'])
    LumiSectionsRange = [LumiSectionsRange_low, LumiSectionsRange_up]
    number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
    output_path = config['DATA']['output_path']
    HLT_name = config['HLT']['HLTname']
    L1_name = config['HLT']['L1name']
    EphemeralFolder = config['DATA']['EphemeralFolder']
    runArea = config['RUNINFO']['Area']
    mode = config['MODE']['Rate_to_compute']
    mode_list = ['L1logic', 'HLTlogic', 'HLTflag']
    if mode not in mode_list:
        raise('Unrecognised mode')

    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(self.number_of_ephemeral_folder):
            branches[i] = self.output_path + f'result_{self.RefRun}_'+self.mode+'/' + 'folder_' + str(i) +'.json'
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        EphemeralFolderName = self.EphemeralFolder + f"EphemeralHLTPhysics{self.branch}_Run{self.runArea}/"
        event_counter = {}
        print(f"-----------------------------------------------Processing folder {self.branch}-----------------------------------------------")
        FileNameList = files_from_path(EphemeralFolderName)
        for FileName in FileNameList:
            if FileName == (self.EphemeralFolder + 'EphemeralHLTPhysics1_Run2023C/nano.tmp'):
                continue
            event_counter[FileName] = {}
            print(f"For {FileName.replace(self.EphemeralFolder, '')}:")
            eph_dataset = Ephemeral(FileName)
            if self.mode == 'HLTflag':
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_flag(run = self.RefRun, lumiSections_range = self.LumiSectionsRange, flagname = self.HLT_name)
            if self.mode == 'HLTlogic':
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(run = self.RefRun, lumiSections_range = self.LumiSectionsRange)
            if self.mode == 'L1lflag':
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_flag(run = self.RefRun, lumiSections_range = self.LumiSectionsRange, flagname = self.L1_name)
            if self.mode == 'L1logic':
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_L1DoubleIsoTau34er2p1(run = self.RefRun, lumiSections_range = self.LumiSectionsRange)
            event_counter[FileName]['N_den'] = N_den_i
            event_counter[FileName]['N_num'] = N_num_i

        self.output().dump(event_counter)
