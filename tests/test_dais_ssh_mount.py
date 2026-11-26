import glob
import os

_TEMPLATES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "slurm_tools"
)
# Templates escape $ so the outer srun bash -c does not expand it.
_SSH_BIND = "--bind {home}/private/cc_ssh_mount:/home/\\$USER/.ssh"


def test_every_dais_apptainer_template_mounts_ssh():
    paths = sorted(glob.glob(os.path.join(_TEMPLATES, "run_slurm_dais_apptainer*.sh")))
    assert paths
    for path in paths:
        with open(path) as handle:
            text = handle.read()
        assert _SSH_BIND in text, path
        assert "apptainer exec" in text, path
