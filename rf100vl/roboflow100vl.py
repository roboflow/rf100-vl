import os
from typing import List, Optional, Iterator
from rf100vl.dataset import RF100VlDataset
from roboflow import Project
import roboflow

def get_rf(api_key: Optional[str] = None):
    if api_key is not None:
        return roboflow.Roboflow(api_key=api_key)
    else:
        ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
        if ROBOFLOW_API_KEY is not None:
            return roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)
        else:
            roboflow.login()
            return roboflow.Roboflow()


class DatasetList:
    def __init__(self, projects: List[Project]):
        self.projects = sorted(projects, key=lambda p: p.name)
        self.datasets = [RF100VlDataset(p) for p in self.projects]

    def __iter__(self) -> Iterator[RF100VlDataset]:
        return iter(self.datasets)

    def download(self, path: str, model_format: str = "coco", overwrite: bool = True) -> None:
        os.makedirs(path, exist_ok=True)
        for dataset in self.datasets:
            dataset.download(os.path.join(path, dataset.name), model_format, overwrite)
    
    def __len__(self) -> int:
        return len(self.datasets)
    
    def __getitem__(self, index: int) -> RF100VlDataset:
        return self.datasets[index]

def get_rf100vl_projects(api_key: Optional[str] = None):
    rf = get_rf(api_key)
    workspace = rf.workspace("rf100-vl")
    projects = []
    for project in workspace.project_list:
        project = Project(api_key=rf.api_key, a_project=project, model_format="coco")
        projects.append(project)

    return DatasetList(projects)


def get_rf100vl_fsod_projects(api_key: Optional[str] = None):
    rf = get_rf(api_key)
    workspace = rf.workspace("rf100-vl-fsod")
    projects = []
    for project in workspace.project_list:
        project = Project(api_key=rf.api_key, a_project=project, model_format="coco")
        projects.append(project)
    return DatasetList(projects)


def get_rf20vl_fsod_projects(api_key: Optional[str] = None) -> DatasetList:
    rf = get_rf(api_key)
    workspace = rf.workspace("rf20-vl-fsod")
    projects = []
    for project in workspace.project_list:
        project = Project(api_key=rf.api_key, a_project=project, model_format="coco")
        projects.append(project)
    return DatasetList(projects)


def get_rf20vl_full_projects(api_key: Optional[str] = None) -> DatasetList:
    rf = get_rf(api_key)
    workspace = rf.workspace("rf20-vl")
    projects = []
    for project in workspace.project_list:
        project = Project(api_key=rf.api_key, a_project=project, model_format="coco")
        projects.append(project)
    return DatasetList(projects)


def download_rf100vl(path: str, model_format: str = "coco", overwrite: bool = True, api_key: Optional[str] = None) -> DatasetList:
    rf100vl_projects = get_rf100vl_projects(api_key)
    rf100vl_projects.download(path, model_format, overwrite)
    return rf100vl_projects


def download_rf100vl_fsod(
    path: str, model_format: str = "coco", overwrite: bool = True, api_key: Optional[str] = None
) -> DatasetList:
    rf100vl_fsod_projects = get_rf100vl_fsod_projects(api_key)
    rf100vl_fsod_projects.download(path, model_format, overwrite)
    return rf100vl_fsod_projects


def download_rf20vl_fsod(path: str, model_format: str = "coco", overwrite: bool = True, api_key: Optional[str] = None) -> DatasetList:
    rf20vl_fsod_projects = get_rf20vl_fsod_projects(api_key)
    rf20vl_fsod_projects.download(path, model_format, overwrite)
    return rf20vl_fsod_projects


def download_rf20vl_full(path: str, model_format: str = "coco", overwrite: bool = True, api_key: Optional[str] = None) -> DatasetList:
    rf20vl_full_projects = get_rf20vl_full_projects(api_key)
    rf20vl_full_projects.download(path, model_format, overwrite)
    return rf20vl_full_projects


# ---------------------------------------------------------------------------
# Single-index helpers — pick the i-th rf100-vl dataset deterministically.
#
# Useful in fan-out training jobs where each pod gets a JOB_COMPLETION_INDEX
# and needs to fetch just one dataset (downloading all 100 in every pod is
# wasteful and flaky). The DatasetList from get_rf100vl_projects() is already
# sorted by name, so `[i]` is stable across runs.
# ---------------------------------------------------------------------------

_PROJECT_FETCHERS = {
    "rf100vl": get_rf100vl_projects,
    "rf100vl_fsod": get_rf100vl_fsod_projects,
    "rf20vl_full": get_rf20vl_full_projects,
    "rf20vl_fsod": get_rf20vl_fsod_projects,
}


def get_rf100vl_dataset_by_index(
    index: int,
    *,
    variant: str = "rf100vl",
    api_key: Optional[str] = None,
) -> "RF100VlDataset":
    """Return the i-th dataset in the chosen rf100-vl variant.

    Parameters
    ----------
    index : int
        Zero-based index, sorted alphabetically by project name.
    variant : str
        One of ``rf100vl`` (default), ``rf100vl_fsod``, ``rf20vl_full``, ``rf20vl_fsod``.
    api_key : str, optional
        Roboflow API key. Falls back to ``ROBOFLOW_API_KEY`` env var.
    """
    fetcher = _PROJECT_FETCHERS.get(variant)
    if fetcher is None:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(_PROJECT_FETCHERS)}"
        )
    datasets = fetcher(api_key)
    n = len(datasets)
    if not 0 <= index < n:
        raise IndexError(f"variant {variant!r} has {n} datasets; index {index} out of range")
    return datasets[index]


def download_rf100vl_index(
    index: int,
    path: str,
    *,
    variant: str = "rf100vl",
    model_format: str = "coco",
    overwrite: bool = True,
    api_key: Optional[str] = None,
) -> "RF100VlDataset":
    """Download just the i-th dataset to ``path``. Returns the materialized dataset."""
    dataset = get_rf100vl_dataset_by_index(index, variant=variant, api_key=api_key)
    dataset.download(path, model_format=model_format, overwrite=overwrite)
    return dataset