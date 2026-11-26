import os

import pytest

from slurm_tools.cluster_layout import layout_for_template
from slurm_tools.do_slurm_job import obtain_parser, slurm_job

MARLLM_DAIS_TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "slurm_tools",
    "run_slurm_dais_apptainer_marllm.sh",
)


def _submit_dry(tmp_path, monkeypatch, extra_argv):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    parser = obtain_parser()
    args = parser.parse_args(
        [
            "--dry",
            "--skip_program_file_check",
            "--program_call",
            "train.py",
            "--template_file",
            MARLLM_DAIS_TEMPLATE,
            "--run_group",
            "g",
            *extra_argv,
        ]
    )
    slurm_job(**vars(args))
    scripts = list((tmp_path / "runs" / "g").glob("*/slurm_script.sh"))
    assert len(scripts) == 1
    return scripts[0].read_text()


def test_wandb_mode_defaults_to_offline(tmp_path, monkeypatch):
    text = _submit_dry(tmp_path, monkeypatch, [])
    assert "--env WANDB_MODE=offline" in text
    assert "--env WANDB_MODE=online" not in text
    secrets = layout_for_template(MARLLM_DAIS_TEMPLATE)["secrets_file"]
    assert f"--env-file {secrets}" in text


def test_wandb_mode_online(tmp_path, monkeypatch):
    text = _submit_dry(tmp_path, monkeypatch, ["--wandb_mode", "online"])
    assert "--env WANDB_MODE=online" in text
    assert "--env WANDB_MODE=offline" not in text
    secrets = layout_for_template(MARLLM_DAIS_TEMPLATE)["secrets_file"]
    assert f"--env-file {secrets}" in text


def test_wandb_mode_rejects_unknown():
    parser = obtain_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dry",
                "--skip_program_file_check",
                "--program_call",
                "train.py",
                "--wandb_mode",
                "online; rm -rf ~",
            ]
        )
