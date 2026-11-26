import os

from slurm_tools.cluster_layout import layout_for_template
from slurm_tools.do_slurm_job import obtain_parser, slurm_job

VIPER_TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "slurm_tools",
    "run_slurm_viper12i_apptainer_marllm.sh",
)


SD_TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "slurm_tools",
    "run_slurm_viper12i_apptainer_social_deduction.sh",
)


def _submit_dry(tmp_path, monkeypatch, extra_argv, template=VIPER_TEMPLATE):
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
            template,
            "--run_group",
            "g",
            "--image",
            "/viper/ptmp2/ykeller/cc/containers/apptainer/marllm_sd_viper.sif",
            "--mem",
            "110000",
            *extra_argv,
        ]
    )
    slurm_job(**vars(args))
    scripts = list((tmp_path / "runs" / "g").glob("*/slurm_script.sh"))
    assert len(scripts) == 1
    return scripts[0].read_text()


def test_viper12i_template_uses_rocm_ib_and_apu_constraint(tmp_path, monkeypatch):
    text = _submit_dry(tmp_path, monkeypatch, [])
    assert '#SBATCH --constraint="apu"' in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --mem=110000" in text
    assert "--rocm" in text
    assert "--nv" not in text
    assert "--bind /etc/hosts:/etc/hosts" in text
    assert "--bind /etc/resolv.conf:/etc/resolv.conf" in text
    assert "--bind /dev/infiniband:/dev/infiniband" in text
    assert "--env HSA_XNACK=1" in text
    assert "--env HIP_VISIBLE_DEVICES=" in text
    assert "--env ROCR_VISIBLE_DEVICES=" in text
    assert "/viper/ptmp2/ykeller/cc/containers/apptainer/marllm_sd_viper.sif" in text
    lay = layout_for_template(VIPER_TEMPLATE)
    assert f"{lay['repo_root']}/MARL-LM:/opt/MARL-LM" in text
    assert "--env WANDB_MODE=offline" in text


def test_viper12i_social_deduction_template_binds_its_own_tree(tmp_path, monkeypatch):
    text = _submit_dry(
        tmp_path,
        monkeypatch,
        [],
        template=SD_TEMPLATE,
    )
    assert '#SBATCH --constraint="apu"' in text
    assert "--rocm" in text
    assert "--nv" not in text
    assert "/opt/data/social_deduction" in text
    lay = layout_for_template(SD_TEMPLATE)
    assert f"{lay['scratch']}/models/social_deduction:/opt/models" in text
    assert f"{lay['repo_root']}/social_deduction_llm:/opt/sh_finetuning" in text
    assert "--pwd /opt/sh_finetuning" in text
    assert "MARL-LM" not in text
    assert "coopbot" not in text
    assert "sh_finetuning_viper.sif" not in text
