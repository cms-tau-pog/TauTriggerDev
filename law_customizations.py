import law
import math
import os
law.contrib.load("htcondor")

class Task(law.Task):
    """
    Base Task
    """
    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
    
class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
    """
    Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
    to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
    Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
    the CERN HTCondor environment. In most cases, like in this example, only a minimal amount of
    configuration is required.
    """
    max_runtime = law.DurationParameter(default=12.0, unit="h", significant=False, description="maximum runtime (default unit is hours). Default: 12h")

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        global_path = os.getenv("RUN_PATH")
        path = os.path.join(global_path, "jobs")
        os.makedirs(path, exist_ok=True)
        return law.LocalDirectoryTarget(path)

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return law.util.rel_path(__file__, "bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        global_path = os.getenv("RUN_PATH")
        # render_variables are rendered into all files sent with a job
        config.render_variables["RUN_PATH"] = global_path
        # force to run on CC7, http://batchdocs.web.cern.ch/batchdocs/local/submit.html#os-choice
        config.custom_content.append(("requirements", "(OpSysAndVer =?= \"CentOS7\")"))
        # maximum runtime
        config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
        # copy the entire environment
        config.custom_content.append(("getenv", "true"))
        log_path = os.path.join(global_path, "logs")
        os.makedirs(log_path, exist_ok=True)
        config.custom_content.append(("log", os.path.join(log_path, 'job.$(ClusterId).$(branches).log')))
        config.custom_content.append(("output", os.path.join(log_path, 'job.$(ClusterId).$(branches).out')))
        config.custom_content.append(("error", os.path.join(log_path, 'job.$(ClusterId).$(branches).err')))
        return config
