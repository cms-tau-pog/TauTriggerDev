#!/usr/bin/env python
import law
import os
import shutil
from dataset import Dataset
from helpers import files_from_path, load_cfg_file, hadd_anatuple
from law_customizations import Task, HTCondorWorkflow

class SaveEventsDenRate(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce root file where Events passing denominator selection are saved 
    '''
    config = load_cfg_file()
    RefRun = int(config['RUNINFO']['ref_run'])
    LumiSectionsRange = [int(config['RUNINFO']['LumiSectionsRange_low']), int(config['RUNINFO']['LumiSectionsRange_up'])]
    number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
    runArea = config['RUNINFO']['Area']
    EphemeralFolder = config['DATA']['SamplesPath']
    output_path = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    tmp_folder = config['DATA']['tmpPath']
    
    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(self.number_of_ephemeral_folder):
            branches[i] = os.path.join(self.output_path, f'EphemeralHLTPhysics{i}_Run{self.runArea}.root')
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        print(f"-----------------------------------------------Processing folder EphemeralHLTPhysics{self.branch}-----------------------------------------------")
        EphemeralFolderName = os.path.join(self.EphemeralFolder, f"EphemeralHLTPhysics{self.branch}_Run{self.runArea}/")
        output_root_file = os.path.join(self.output_path, f'EphemeralHLTPhysics{self.branch}_Run{self.runArea}.root')
        output_tmp_folder = os.path.join(self.tmp_folder, f"EphemeralHLTPhysics{self.branch}_Run{self.runArea}")

        if os.path.exists(output_tmp_folder):
            print(f'A tmp folder which store tmp root files exist already: being deleted')
            shutil.rmtree(output_tmp_folder)
        os.makedirs(output_tmp_folder)

        # Produce tmp files
        FileNameList = files_from_path(EphemeralFolderName)
        for FileName in FileNameList:
            # this tmp folder fuck up the code
            if FileName == os.path.join(self.EphemeralFolder, 'EphemeralHLTPhysics1_Run2023C', 'nano.tmp'):
                continue
            print(f"Producing tmp file for {os.path.basename(FileName)}:")
            output_tmp_file = os.path.join(output_tmp_folder, os.path.basename(FileName))
            eph_dataset = Dataset(FileName)
            eph_dataset.Save_Event_Nden_Rate(output_tmp_file, self.RefRun, self.LumiSectionsRange)

        # Hadd the tmp files to a single root file
        hadd_anatuple(output_tmp_folder, output_root_file)
        shutil.rmtree(output_tmp_folder)
        return
    
class ProduceRateFiles(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce json files where number of events which pass denominator and numerator criteria are saved from each ephemerals files
    '''
    config = load_cfg_file()
    RefRun = int(config['RUNINFO']['ref_run'])
    number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
    HLT_name = config['HLT']['HLTname']
    runArea = config['RUNINFO']['Area']
    PNet_treshold = config['OPT']['PNet_treshold']
    intput_root = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    output_path = os.path.join(config['DATA']['result_rate'], f'result_{RefRun}')

    if PNet_treshold == 'None':
        PNetConfig = False
        output_path = os.path.join(output_path, HLT_name)
    else:
        PNetConfig = True
        PNetTreshold = float(PNet_treshold)
        output_path = os.path.join(output_path, f'PNetTresh_{PNetTreshold}')

    # requires SaveEventsDenRate for Den events
    def workflow_requires(self):
        return { "Counter": SaveEventsDenRate.req(self, branch=self.branch) }

    def requires(self):
        return self.workflow_requires()
    
    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(self.number_of_ephemeral_folder):
            branches[i] = os.path.join(self.output_path, f'EphemeralHLTPhysics{i}_Run{self.runArea}.json')
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        print(f"-----------------------------------------------Processing folder EphemeralHLTPhysics{self.branch}_Run{self.runArea}.root-----------------------------------------------")

        intput_root_file = os.path.join(self.intput_root, f'EphemeralHLTPhysics{self.branch}_Run{self.runArea}.root')

        event_counter = {}
        event_counter[intput_root_file] = {}
        print(f"For {os.path.basename(intput_root_file)}:")
        eph_dataset = Dataset(intput_root_file)
        if self.PNetConfig:
            N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_DiTauPNet(treshold = self.PNetTreshold)
        else:
            N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()

        event_counter[intput_root_file]['N_den'] = N_den_i
        event_counter[intput_root_file]['N_num'] = N_num_i

        self.output().dump(event_counter)
