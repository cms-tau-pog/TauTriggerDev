# Study: use of Pnet information at HLT level

integration of ParticleNet algorithm for tau-identification into HLT

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
- Install and activate the environment that I named "HNL" with the command:
```shell
conda env create -f environment.yml
```

### General LAW commands

This code is using law (luigi analysis workflow) to produce files when there is a need to process a lot of events from root files.
This is some useful commands that you can use with law

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
A task can have multiple branches to parallelize jobs, so here you'll run a task locally for the first branch (0).

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
Make sure you have the correct path to your repository in env.sh

- Create an index file, too list all the tasks 
```shell
law index --verbose
```

Then depending on what you want to do:

### ComputeRate

Compute rate (of HLTs, L1, ...) of a run, computed with Ephemeral dataset. Then you can compare from nominal values in CMS OMS. All parameters must be set in ComputeRate/config.ini. Rate_to_compute parameter allows you to specify which rate you want to compute. Some parameters of the run you want to study must be set using information from CMS OMS.

First you need to produce files with ProduceRateFiles tasks:
- ProduceRateFiles: Produce files that can be used in order to compute rate, by saving the number of events that pass numerator and denominator. Output are json files (1 by ephemeral folder). N_den is the number of event that are in ref_run and within LumiSectionsRange. N_num is the number of event that are in ref_run and within LumiSectionsRange that pass certain conditions (depending on the mode Rate_to_compute you choose)
```shell
law run ProduceRateFiles
```

Then you can compute the rate using L1A_physics rate:
```shell
python ComputeRate/compute_rate.py
```

## Documentation
- Akward array: https://awkward-array.readthedocs.io/en/latest/index.html
- law documentation: https://luigi.readthedocs.io/en/stable/#
- Example that demonstrates how to create law task workflows that run on the HTCondor batch system at CERN: https://github.com/riga/law/tree/master/examples/htcondor_at_cern
- CMS OMS: https://twiki.cern.ch/twiki/bin/view/Main/OnlineMonitoringSystemOMSTutorial
- HLT confDB: https://hlt-config-editor-dev-confdbv3.app.cern.ch/open?cfg=%2Ffrozen%2F2023%2F2e34%2Fv1.1%2FHLT%2FV1&db=offline-run3
- CMS OMS L1 rate: https://cmsoms.cern.ch/cms/triggers/
- Optimal run to choose: https://indico.cern.ch/event/1277725/contributions/5367044/attachments/2661315/4610455/TauTriggerMeeting07June2023.pdf
