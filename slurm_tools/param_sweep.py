from slurm_tools.do_slurm_job import slurm_job, obtain_parser
import itertools
from typing import Any
from slurm_tools.util import load_yaml
import json


def format_param_choices(param_choices: dict[str, Any]):
    return " ".join([f"--{key} {value}" for key, value in param_choices.items()])


def param_sweep_grid(sweep_params: dict[str, list], program_call: str, **job_kwargs):
    """
    Run a parameter sweep over a grid of hyperparameters.

    Args:
        sweep_params (dict[str, list]): A dictionary mapping hyperparameter names to lists of values to sweep over.
        **job_kwargs: Keyword arguments to pass to slurm_job.

    Returns:
        None
    """
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    for param_values in itertools.product(*values):
        param_choices = dict(zip(keys, param_values))
        choice_program_call = f"{program_call} {format_param_choices(param_choices)}"

        slurm_job(program_call=choice_program_call, **job_kwargs)


def main():
    parser = obtain_parser()
    parser.add_argument("--sweep_config_path", type=str, default="")
    parser.add_argument("--sweep_params", type=str, default="")
    args = parser.parse_args()

    assert (
        args.sweep_config_path or args.sweep_params
    ), "No sweep configuration provided."

    sweep_config = load_yaml(args.sweep_config_path)
    if args.sweep_params:
        sweep_config.update(json.loads(args.sweep_params))

    delattr(args, "sweep_config_path")
    delattr(args, "sweep_params")

    param_sweep_grid(sweep_config, **vars(args))
