# Installation
`pip install -e .`

# Usage
Start a single slurm job. Currently, only slurm jobs with conda envs are tested.
```bash
usage: slurm_job [-h] [--dry] [--n_gpu N_GPU] [--time TIME] [--template_file TEMPLATE_FILE] [--run_group RUN_GROUP] --program_call PROGRAM_CALL [--image IMAGE] [--launcher LAUNCHER]
                 [--acc_config ACC_CONFIG] [--n_nodes N_NODES] [--conda_env CONDA_ENV]
```
For distributed training, set launcher to `torchrun` (Tested) or `accelerate` (untested).

Check the progress of your slurm job:
```bash
usage: monitor_run [-h] [--watch] [--watchtime WATCHTIME] [--oldness OLDNESS] [--path PATH] [--script]
```

Start a parameter sweep:
```bash
usage: param_sweep [-h] [--dry] [--n_gpu N_GPU] [--time TIME] [--template_file TEMPLATE_FILE] [--run_group RUN_GROUP] --program_call PROGRAM_CALL [--image IMAGE] [--launcher LAUNCHER]
                   [--acc_config ACC_CONFIG] [--n_nodes N_NODES] [--conda_env CONDA_ENV] [--sweep_config_path SWEEP_CONFIG_PATH] [--sweep_params SWEEP_PARAMS]
```