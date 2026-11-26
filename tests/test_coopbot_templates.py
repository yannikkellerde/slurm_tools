import os

from slurm_tools.cluster_layout import layout_for_template
from slurm_tools.do_slurm_job import obtain_parser, slurm_job

_TEMPLATES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "slurm_tools"
)
RAVEN = os.path.join(_TEMPLATES, "run_slurm_raven_apptainer_coopbot.sh")
ARRHENIUS = os.path.join(_TEMPLATES, "run_slurm_arrhenius_apptainer_coopbot.sh")
VIPER = os.path.join(_TEMPLATES, "run_slurm_viper12i_apptainer_coopbot.sh")
DAIS = os.path.join(_TEMPLATES, "run_slurm_dais_apptainer_coopbot.sh")

COOPBOT_RAVEN = "/raven/ptmp/ykeller/cc/containers/apptainer/coopbot.sif"
COOPBOT_ARRHENIUS = (
    "/nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik"
    "/containers/apptainer/coopbot.sif"
)
COOPBOT_VIPER = "/viper/ptmp2/ykeller/cc/containers/apptainer/coopbot.sif"
COOPBOT_DAIS = "/dais/fs/scratch/ykeller/containers/apptainer/coopbot.sif"


def _submit_dry(tmp_path, monkeypatch, template, image, extra_argv):
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
            image,
            *extra_argv,
        ]
    )
    slurm_job(**vars(args))
    scripts = list((tmp_path / "runs" / "g").glob("*/slurm_script.sh"))
    assert len(scripts) == 1
    return scripts[0].read_text()


def test_raven_coopbot_template_binds_and_image(tmp_path, monkeypatch):
    text = _submit_dry(tmp_path, monkeypatch, RAVEN, COOPBOT_RAVEN, ["--mem", "0"])
    assert "#SBATCH --gres=gpu:a100:1" in text
    assert "--nv" in text
    assert COOPBOT_RAVEN in text
    lay = layout_for_template(RAVEN)
    assert f"{lay['repo_root']}/llm-strategic-tuning:/opt/llm-strategic-tuning" in text
    assert f"{lay['scratch']}/data/coopbot:/opt/data/coopbot" in text
    assert "--pwd /opt/llm-strategic-tuning" in text
    assert "MARL-LM" not in text
    assert "--env WANDB_MODE=offline" in text


def test_arrhenius_coopbot_template_binds_and_image(tmp_path, monkeypatch):
    text = _submit_dry(
        tmp_path,
        monkeypatch,
        ARRHENIUS,
        COOPBOT_ARRHENIUS,
        ["--mem", "0", "--account", "ehpc-dev-2026d09-182-gpu"],
    )
    assert "#SBATCH --gres=gpu:nvidia_gh200_120gb:1" in text
    assert COOPBOT_ARRHENIUS in text
    assert (
        "/nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik"
        "/github_repos/llm-strategic-tuning:/opt/llm-strategic-tuning"
    ) in text
    assert "--pwd /opt/llm-strategic-tuning" in text
    assert "MARL-LM" not in text
    assert "--env SLINGSHOT_VNIS=" in text


def test_viper12i_coopbot_template_uses_rocm_and_coopbot_sif(tmp_path, monkeypatch):
    text = _submit_dry(tmp_path, monkeypatch, VIPER, COOPBOT_VIPER, ["--mem", "110000"])
    assert '#SBATCH --constraint="apu"' in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "--rocm" in text
    assert "--nv" not in text
    assert "--bind /etc/hosts:/etc/hosts" in text
    assert "--bind /etc/resolv.conf:/etc/resolv.conf" in text
    assert COOPBOT_VIPER in text
    lay = layout_for_template(VIPER)
    assert f"{lay['repo_root']}/llm-strategic-tuning:/opt/llm-strategic-tuning" in text
    assert "--env HSA_XNACK=1" in text
    assert "MARL-LM" not in text


def test_dais_coopbot_template_has_cluster_sync(tmp_path, monkeypatch):
    text = _submit_dry(tmp_path, monkeypatch, DAIS, COOPBOT_DAIS, ["--mem", "120G"])
    assert COOPBOT_DAIS in text
    assert "--bind /etc/hosts:/etc/hosts" in text
    assert "--bind /etc/resolv.conf:/etc/resolv.conf" in text
    lay = layout_for_template(DAIS)
    assert f"{lay['scratch']}/cluster_sync:/opt/cluster_sync" in text
    assert f"{lay['repo_root']}/llm-strategic-tuning:/opt/llm-strategic-tuning" in text
    assert f"--bind {lay['home']}/private/cc_ssh_mount:/home/\\$USER/.ssh" in text
    assert "--pwd /opt/llm-strategic-tuning" in text
