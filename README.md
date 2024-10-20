# Installation
`pip install -e .`

# Goal
Easy way to start and debug slurm jobs using conda on one or muliple nodes / gpus.

## What you can do with slurm without this tool
+ Get a one GPU debugging shell `srun -p gpu --cpus-per-task=18 --gres=gpu:a100:1 --mem=0 --time=01:00:00 --pty bash`
+ Create an sbatch file manually and run it on multiple gpus `sbatch path/to/sbatch/script`
+ Check your running jobs `squeue --me`
+ Cancel jobs `scancel {jobid}`
+ Attach shell to running job `srun --pty --overlap --jobid {jobid} bash`

## Typical troubles while using slurm without this tool
+ You want to test/debug your multi-gpu application. 
    1. You write an sbatch script for it.
    2. You submit the job.
    3. It takes 10 min for slurm to allocate gpus for your job.
    4. Then your job fails immediately because of a typo.
    5. You fix the typo in 5 seconds and resubmit the job. It takes 30 min this time for slurm to allocate the gpus.
    6. Your debuggig process is very slow  

+ You want to work on a new project.
    - In addition to the project code, you'll have to design a new sbatch file and probably include it in your repo.
    - Each time you want to change run parameters such as memory, gpu, time or conda env, you'll have to change the sbatch file.
    
+ You want to keep good track of the outputs of your slurm jobs.
    - To monitor the output of your run, you have to find and cat the output log file.
    - Log files quickly become unorganized
    - You may forget what parameters / sbatch script each job was started when coming back to it.

# Solutions
+ Quickly start batch jobs for new project.
    - Just make sure to install slurm-tools. Then start a new job for your task using the `slurm_job` utility.
    - Use the `--program_call` parameter to specify the python file and arguments.
    - Use `--launcher python` for single gpu tasks or `--launcher torchrun` for muliti gpu/node tasks.
    - Example command: `slurm_job --n_gpu 4 --time 00:20:00 --launcher torchrun --n_nodes 2 --program_call "main.py --config default.yml" --conda_env my_conda_env` 
    - Check more parameters using `slurm_job --help`
    
+ Track outputs of slurm jobs.
    - Use the `monitor_run` utility. It will show the outputs of the most recent slurm_job.
    - Add `--watch` to follow it continually.
    - Outputs, scripts and redo files are logged in your `~/runs` folder
    
+ Debug multi-gpu application
    - Add `--keepalive 400` to your `slurm_job` parameters. Your slurm_job now won't stop for 400 seconds after your program_call fails.
    - Monitor your slurm job using the `monitor_slurm` utility. If your code fails or stops, you now have 400 seconds to restart the program using the same slurm job / gpu allocation.
    - Use the `attach` utility to attach a shell to your slurm job.
    - Call `redo` to get the path to your redo file. You may edit the contents of that file to call your program in a slightly different way.
    - Run `redo --execute` to rerun your job using the redo file.
    - Your slurm_job won't stop as long you are running a python application plus 400 seconds.
    - You can monitor the outputs of the redo using the `monitor_slum` tool (just like any other job).
    - You can also set the keepalive value to different values. However, be aware that you are risking blocking the gpus without using them for keepalive seconds. So to treat the HPC clusters resources well, use only low keepalive values. Especially if you are using a lot of gpus.
    
# Experimental
+ Parameter sweep
    - Run a simple grid search over some parameters of your script.
    - Use the `param_sweep` cl utility. Specify the grid of parameters to serach either directly via the `--sweep_params` parameter or using a yaml file and the `--sweep_config_path` parameter.
    - If your `main.py` script takes parameter `lr` and `batch_size`, a param sweep may be started like this:
        * `param_sweep --n_gpu 1 --time 00:20:00 --program_call "main.py" --conda_env my_conda_env --sweep_params "{'batch_size':[32, 64], 'lr':[1e-4,1e-5]}"`
+ Apptainer support
    - For now, only conda is really supported. But there is `run_slurm_apptainer.sh` which may be a good starting point to get this tool to work with apptainer.
+ Accelerate as a launcher support
    - For now, only torchrun is supported. But accelerate was supported by this codebase once, so it should not be too hard to make it work again.
