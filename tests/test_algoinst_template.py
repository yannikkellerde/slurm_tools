import os

import pytest

from slurm_tools.cluster_layout import layout_for_template
from slurm_tools.do_slurm_job import obtain_parser, slurm_job

_TEMPLATES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "slurm_tools"
)
RAVEN = os.path.join(_TEMPLATES, "run_slurm_raven_apptainer_algoinst.sh")
ALGOINST_RAVEN = "/raven/ptmp/ykeller/cc/containers/apptainer/algoinst.sif"


def test_raven_algoinst_template_binds_image_and_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    parser = obtain_parser()
    args = parser.parse_args(
        [
            "--dry",
            "--skip_program_file_check",
            "--program_call",
            "-m aimanager train-ah configs/training/example.yml",
            "--template_file",
            RAVEN,
            "--run_group",
            "g",
            "--image",
            ALGOINST_RAVEN,
            "--mem",
            "16G",
            "--launcher",
            "python",
        ]
    )
    slurm_job(**vars(args))
    scripts = list((tmp_path / "runs" / "g").glob("*/slurm_script.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    assert "#SBATCH --gres=gpu:a100:1" in text
    assert "--nv" in text
    assert ALGOINST_RAVEN in text
    lay = layout_for_template(RAVEN)
    assert (
        f"{lay['repo_root']}/algorithmic-institutions:/opt/algorithmic-institutions"
        in text
    )
    assert "--pwd /opt/algorithmic-institutions" in text
    assert "--env WANDB_ENTITY=chm-ml" in text
    assert "--env WANDB_PROJECT=algorithmic-institutions" in text
    assert "--env WANDB_MODE=offline" in text
    assert "python -m aimanager train-ah" in text
    assert "coopbot" not in text
    assert "MARL-LM" not in text
    assert "cc-snapshots" not in text


def test_raven_algoinst_checkout_binds_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    parser = obtain_parser()
    args = parser.parse_args(
        [
            "--dry",
            "--skip_program_file_check",
            "--program_call",
            "-m aimanager simulate configs/simulation/example.yml",
            "--template_file",
            RAVEN,
            "--run_group",
            "g",
            "--image",
            ALGOINST_RAVEN,
            "--mem",
            "16G",
            "--launcher",
            "python",
            "--checkout",
            sha,
        ]
    )
    slurm_job(**vars(args))
    scripts = list((tmp_path / "runs" / "g").glob("*/slurm_script.sh"))
    assert len(scripts) == 1
    text = scripts[0].read_text()
    lay = layout_for_template(RAVEN)
    home, repo = lay["home"], lay["repo_root"]
    assert (
        f"--bind {home}/cc-snapshots/{sha}:/opt/algorithmic-institutions:ro"
        in text
    )
    assert (
        f"--bind {repo}/algorithmic-institutions:/opt/algorithmic-institutions"
        not in text
    )
    assert (
        f"--bind {repo}/algorithmic-institutions/artifacts:"
        "/opt/algorithmic-institutions/artifacts:ro" in text
    )
    assert (
        f"--bind {home}/cc-snapshot-outputs/{sha}/plots:"
        "/opt/algorithmic-institutions/plots" in text
    )
    assert (
        f"--bind {home}/cc-snapshot-outputs/{sha}/.log:"
        "/opt/algorithmic-institutions/.log" in text
    )
    assert f"mkdir -p {home}/cc-snapshot-outputs/{sha}/plots" in text
    assert f"snapshot {sha} is not on the cluster" in text


def test_checkout_rejects_a_branch_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    parser = obtain_parser()
    args = parser.parse_args(
        [
            "--dry",
            "--skip_program_file_check",
            "--program_call",
            "-m aimanager simulate c.yml",
            "--template_file",
            RAVEN,
            "--run_group",
            "g",
            "--image",
            ALGOINST_RAVEN,
            "--checkout",
            "main",
        ]
    )
    with pytest.raises(ValueError, match="40-character"):
        slurm_job(**vars(args))
