#!/usr/bin/env python
import law
import os
from ComputeRate.Ephemeral_dataset import Ephemeral
from helpers import files_from_path, load_cfg_file_ComputeRate
from law_customizations import Task, HTCondorWorkflow
import configparser

class ComputeRate(Task, HTCondorWorkflow, law.LocalWorkflow):
    config = load_cfg_file_ComputeRate()
    RefRun = int(config['RUNINFO']['ref_run'])
    LumiSectionsRange_low = int(config['RUNINFO']['LumiSectionsRange_low'])
    LumiSectionsRange_up = int(config['RUNINFO']['LumiSectionsRange_up'])
    LumiSectionsRange = [LumiSectionsRange_low, LumiSectionsRange_up]
    number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
    output_path = config['DATA']['output_path']
    HLT_name = config['HLT']['HLTname']
    EphemeralFolder = config['DATA']['EphemeralFolder']

    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(self.number_of_ephemeral_folder):
            branches[i] = self.output_path + 'folder_' + str(i) +'.json'
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):


        EphemeralFolderName = self.EphemeralFolder + f"EphemeralHLTPhysics{self.branch}_Run2022G/"
        event_counter = {}
        print(f"-----------------------------------------------Processing folder {self.branch}-----------------------------------------------")
        FileNameList = files_from_path(EphemeralFolderName)
        for FileName in FileNameList:
            event_counter[FileName] = {}
            print(f"For {FileName.replace(self.EphemeralFolder, '')}:")
            eph_dataset = Ephemeral(FileName)
            N_den_i, N_num_i = eph_dataset.compute_rate(run = self.RefRun, lumiSections_range = self.LumiSectionsRange, HLTname = self.HLT_name)
            event_counter[FileName]['N_den'] = N_den_i
            event_counter[FileName]['N_num'] = N_num_i

        self.output().dump(event_counter)
