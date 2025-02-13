# Installation
`pip install -e .`

# Goal
Easy way to start and debug slurm jobs using conda on one or muliple nodes / gpus.

## Main usage
+ Quickly start a batched slurm job: `slurm_job --n_gpu 4 --time 00:20:00 --launcher torchrun --n_nodes 2 --program_call "main.py --config default.yml" --conda_env my_conda_env`
    - Select launcher `python` for single GPU jobs and `torchrun` for multi-gpu DDP jobs
    - Easy integration with conda/miniforge by setting conda_env
    - Example for single gpu job: `slurm_job --n_gpu 1 --time 2:30:00 --launcher python --n_nodes 1 --program_call "script.py --config hey.yml" --mem 120GB --conda_env sh_finetune_new`
+ Easily monitor the logs of the recently started jobs: `monitor_run --oldness 0`

## More essential slurm knowledge
+ monitor your slurm jobs `squeue --me`
+ Attach shell to running job `srun --pty --overlap --jobid 15588005 bash` (find jobid via squeue)
+ Get a GPU shell `srun -p gpu --gres=gpu:a100:1 --mem=120GB --time=05:30:00 --pty bash`
+ Cancel a job `scancel 15588005` (find jobid via squeue)

## Typical troubles while using slurm without this tool
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
