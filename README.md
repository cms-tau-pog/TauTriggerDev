# Study: optimisation of tau-identification into HLT

Integration of ParticleNet algorithm for tau-identification into HLT.

##  Requirements

You only need a CERN account, with access to DAS: https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookLocatingDataSamples

###  Load the code

- Clone this repositiry 
```shell
git clone git@github.com:pauldebryas/PnetAtHLT.git
cd PnetAtHLT/
```

### Packages 

Here is a non-exhaustive list of the packages that I used:
- python3
- law
- coffea
- akward array
- root
- numpy
- matplotlib
- yaml

For simplicity, you can use the same conda environment as I, which is saved in a .yml file.
For that:
- Make sure you have conda/miniconda installed: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
- Install and activate the environment that I named "PnetEnv" with the command:
```shell
conda env create -f PnetEnv.yml
```

### General LAW commands

This code is using law (luigi analysis workflow) to produce files when there is a need to process a lot of events from root files.
In law_customizations.py, you can modify the base Task (Task class). You can also modify parameters used when submitting jobs with HTCondor (HTCondorWorkflow class).
Note that for HTcondor submission, bootstrap.sh file is executed prior to the job in order to setup the environment.
This is some useful commands that you can use with law:

- Create an index file, too list all the tasks:
```shell
law index --verbose
```

- Print the status of a task:
```shell
law run name_of_the_task --print-status -1
```

- Run task locally (useful for debugging): 
```shell
law run name_of_the_task --name_of_the_task-workflow local --branch 0
```
A task can have multiple branches to parallelize jobs, so here you will run a task locally for the first branch (0).

- Run task with HTcondor:
```shell
law run name_of_the_task
```

- If you want to limit the number of jobs running simultaneously (here max 100 jobs simultaneously):
```shell
law run name_of_the_task --name_of_the_task-parallel-jobs 100
```

##  Run the code

- Setup the environment (to do at each login)
```shell
source env.sh
```
Make sure you have the correct path to your conda installation in this file.

- Create an index file (.law/index), that list all the tasks needed to run the code (to do each time you add a task or modify dependencies).
```shell
law index --verbose
```
If you add a task, don't forget to put it in law.cfg before running this command.

You should see
```shell
module 'ComputeRate.tasks', 2 task(s):
    - SaveEventsDenRate
    - ProduceRateFiles

module 'ComputeEfficiency.tasks', 2 task(s):
    - SaveEventsDenEfficiency
    - ProduceEfficiencyFiles
```
The tasks, written in ComputeRate/tasks.py and in ComputeEfficiency/tasks.py are listed here.

This code allows you to compute rate and efficiency of a HLT path. All parameters must be set in config.ini, and description is provided in the file. You can also find some example of config files in config_files/ folder.
Logic of HLT path can be found in confDB and are scripted in python in HLTClass/ folder:
- Dataset: Base class inherited by HLT path
- DiTauDataset: HLT path where there is at least 2 taus
- SingleTauDataset: HLT path where there is at least 1 tau
- DoubleORSingleTauDataset: logic OR between single and double tau path

Optimisation of those HLT is done by replacing DeepTau with PNet for tau identification. An option in the config file (PNet_mode) allows you to switch from baseline HLT (with DeepTau) to the one using PNet.
Since both of them are ratio, the worflow is the following: 
- Root files where all events are stored (from DAS)
--> Save events passing denominator selection (SaveEventsDen... tasks)
- Root files of events passing denominator selection with only the necessary branches used in numerator selection/plotting
--> Save events passing numerator selection (Produce...Files tasks)
- Files of events passing numerator selection 

You can submit HTCondor jobs with law to produce those files. Output of the jobs (err/log/out) can be found in logs/ folder and submitted scripts in jobs/ folder.
More specific informations for rate and efficiency computation in the following parts.

### Compute Rate

Compute rate of HLT for a specifc run, computed with Ephemeral dataset.

To compute the rate you need:
- First to produce files with SaveEventsDenRate tasks: Produce root files (in RateDenPath) by saving only the events that are in the run/lumisection range.
```shell
law run SaveEventsDenRate
```

- Then produce json files (1 by ephemeral folder) that save the number of events which pass the selection of the HLT_name parameter (N_num), and the number of events that are in the run/lumisection range (N_den).
```shell
law run ProduceRateFiles
```

Then you can compute the rate with (N_num/N_den)*L1A_physics rate:
```shell
python ComputeRate/compute_rate.py
```

Since ProduceRateFiles task can take long, you can quickly compute rate with only 1 Ephemeral data file to give an aproximation of the rate:
Then you can compute the rate with (N_num/N_den)*L1A_physics rate:
```shell
python ComputeRate/compute_rate_quick.py
```

### Compute Efficiency

Compute efficiency of the HLT using MC dataset.
You can specify the list of MC processes you want to analyse in the config file (efficiency may differ depending on the MC process used to produce MC data).
To compute the efficiency you need:
- First to produce files with SaveEventsDenEfficiency tasks: Produce root files (in EffDenPath) by saving only the events that match certain conditions (for example at least 2 hadronic taus for DiTau HLT analysis)
```shell
law run SaveEventsDenEfficiency
```

- Then produce root files with ProduceEfficiencyFiles tasks: save the taus in the events which have pass denominator selection, and the ones (subpart) which have pass the selection of the HLT_name parameter (same selection as for rate computation).
```shell
law run ProduceEfficiencyFiles
```

Then you can compute the efficiency with N_num/N_den:
```shell
python ComputeEfficiency/compute_eff.py
```

You can also compute algorithm efficiency, where the denominator selection is more restrictive, so that the only difference between N_num and N_den is due to PNet/DeepTau selection.
```shell
python ComputeEfficiency/compute_eff_algo.py
```

### Optimisation of PNet parameters

In the config file, you can tune some parameters used during PNet selection. Optimisation of PNet selection is dobe by maximisation of signal efficiency, while keeping the rate under Run 3 budget for selection using DeepTau. You can find in Optimisation/ folder some BFGS Minimisation method in order to do that.

### plots

You can find some scripts to produce plots in plot_figures/ folder.

## Documentation
- Akward array: https://awkward-array.readthedocs.io/en/latest/index.html
- law documentation: https://luigi.readthedocs.io/en/stable/#
- Example that demonstrates how to create law task workflows that run on the HTCondor batch system at CERN: https://github.com/riga/law/tree/master/examples/htcondor_at_cern
- CMS OMS: https://twiki.cern.ch/twiki/bin/view/Main/OnlineMonitoringSystemOMSTutorial
- HLT confDB: https://hlt-config-editor-dev-confdbv3.app.cern.ch/open?cfg=%2Ffrozen%2F2023%2F2e34%2Fv1.1%2FHLT%2FV1&db=offline-run3
- CMS OMS L1 rate: https://cmsoms.cern.ch/cms/triggers/
- Optimal run to choose: https://indico.cern.ch/event/1277725/contributions/5367044/attachments/2661315/4610455/TauTriggerMeeting07June2023.pdf
- ParticleNet: https://arxiv.org/abs/1902.08570