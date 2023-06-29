#!/usr/bin/env python
import law
import os
import shutil
from dataset import Dataset
from helpers import files_from_path, load_cfg_file, hadd_anatuple
from law_customizations import Task, HTCondorWorkflow

class SaveEventsDenEfficiency(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce root file where Events passing denominator selection are saved 
    '''
    config = load_cfg_file()
    MCDataPath = config['DATA']['SamplesPath']
    tmp_folder = config['DATA']['tmpPath']
    HLT_name = config['HLT']['HLTname']
    MCDataFolderNames = list(filter(None, (x.strip() for x in config['DATA']['MCDataFolderNames'].splitlines())))
    output_path = os.path.join(config['DATA']['AnatuplePath'], HLT_name)
    
    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(len(self.MCDataFolderNames)):
            branches[i] = os.path.join(self.output_path, f'{self.MCDataFolderNames[i]}.root')
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        print(f"-----------------------------------------------Processing folder {self.MCDataFolderNames[self.branch]}-----------------------------------------------")
        MCFolderName = os.path.join(self.MCDataPath, self.MCDataFolderNames[self.branch])
        output_root_file = os.path.join(self.output_path, f'{self.MCDataFolderNames[self.branch]}.root')
        output_tmp_folder = os.path.join(self.tmp_folder, self.MCDataFolderNames[self.branch])
        if os.path.exists(output_tmp_folder):
            print(f'A tmp folder which store tmp anatuple files exist already: being deleted')
            shutil.rmtree(output_tmp_folder)
        os.makedirs(output_tmp_folder)

        # Produce tmp files
        FileNameList = files_from_path(MCFolderName)
        for FileName in FileNameList:
            print(f"Producing tmp file for {os.path.basename(FileName)}:")
            output_tmp_file = os.path.join(output_tmp_folder, os.path.basename(FileName))
            MC_dataset = Dataset(FileName)
            MC_dataset.save_Event_Nden(output_tmp_file)

        # Hadd the tmp files to a single root file
        hadd_anatuple(output_tmp_folder, output_root_file)
        shutil.rmtree(output_tmp_folder)
        return

class ProduceEfficiencyFiles(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce root file where informations of events that pass numerator/denominator selection are saved
    '''
    config = load_cfg_file()
    HLT_name = config['HLT']['HLTname']
    input_path = os.path.join(config['DATA']['AnatuplePath'], HLT_name)
    MCDataFolderNames = list(filter(None, (x.strip() for x in config['DATA']['MCDataFolderNames'].splitlines())))
    PNet_treshold = config['OPT']['PNet_treshold']
    if PNet_treshold == 'None':
        PNetConfig = False
        output_path = os.path.join(config['DATA']['result_eff'], HLT_name)
    else:
        PNetConfig = True
        PNetTreshold = float(PNet_treshold)     
        output_path = os.path.join(config['DATA']['result_eff'], f'PNetTresh_{PNetTreshold}')

    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(len(self.MCDataFolderNames)):
            branches[i] = os.path.join(self.output_path, f'{self.MCDataFolderNames[i]}.root')
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        print(f"-----------------------------------------------Processing folder {self.MCDataFolderNames[self.branch]}-----------------------------------------------")
        input_root_file = os.path.join(self.input_path, f'{self.MCDataFolderNames[self.branch]}.root')
        output_root_file = os.path.join(self.output_path, f'{self.MCDataFolderNames[self.branch]}.root')

        if not os.path.exists(input_root_file):
            raise('Input root file does not exist')

        MC_dataset = Dataset(input_root_file)
        if self.PNetConfig:
            MC_dataset.produceRoot_DiTauPNet(output_root_file, self.PNetTreshold)
        else:
            MC_dataset.produceRoot_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(output_root_file)

