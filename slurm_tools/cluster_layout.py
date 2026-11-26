"""Account paths written into sbatch scripts.

The command server exports ``CC_CLUSTER_USER``, ``CC_HOME``, ``CC_SCRATCH``,
``CC_REPO_ROOT``, and ``CC_SECRETS_FILE`` before ``slurm_job``. Without
those, the submitting account (``$USER``) and the usual DAIS / Raven /
Viper layout are used. Container images are chosen separately and stay
on the shared image tree.
"""

import os

_RAVEN_REPOS = (
    "MARL-LM",
    "social_deduction_llm",
    "llm-strategic-tuning",
    "algorithmic-institutions",
)


def cluster_user() -> str:
    return os.environ.get("CC_CLUSTER_USER") or os.environ.get("USER") or "unknown"


def layout_for_template(template_file: str) -> dict:
    """Return home, scratch, repo_root, and secrets_file for a template."""
    user = cluster_user()
    home = os.environ.get("CC_HOME") or f"/u/{user}"
    name = os.path.basename(template_file)
    if "viper" in name:
        scratch = os.environ.get("CC_SCRATCH") or f"/viper/ptmp2/{user}/cc"
        repo_root = os.environ.get("CC_REPO_ROOT") or f"{home}/github_repos"
    elif "dais" in name:
        scratch = os.environ.get("CC_SCRATCH") or f"/dais/fs/scratch/{user}"
        repo_root = os.environ.get("CC_REPO_ROOT") or f"{home}/github_repos"
    elif "raven" in name:
        scratch = os.environ.get("CC_SCRATCH") or f"/ptmp/{user}/cc"
        repo_root = os.environ.get("CC_REPO_ROOT") or home
    else:
        scratch = os.environ.get("CC_SCRATCH") or home
        repo_root = os.environ.get("CC_REPO_ROOT") or home
    secrets = os.environ.get("CC_SECRETS_FILE") or f"{home}/private/cc_secrets"
    return {
        "user": user,
        "home": home,
        "scratch": scratch,
        "repo_root": repo_root,
        "secrets_file": secrets,
    }


def layout_for_cluster(cluster: str) -> dict:
    names = {
        "raven": "run_slurm_raven.sh",
        "dais": "run_slurm_dais.sh",
        "viper12i": "run_slurm_viper12i.sh",
    }
    if cluster not in names:
        raise ValueError(f"no layout for cluster {cluster!r}")
    return layout_for_template(names[cluster])
