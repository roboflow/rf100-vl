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
        self.projects = projects
        self.projects = sorted(self.projects, key=lambda p: p.name)
        self.datasets = [RF100VlDataset(p) for p in projects]

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
        project = Project(api_key=api_key, a_project=project, model_format="coco")
        projects.append(project)
    return DatasetList(projects)


def get_rf20vl_fsod_projects(api_key: Optional[str] = None) -> DatasetList:
    rf = get_rf(api_key)
    workspace = rf.workspace("rf20-vl-fsod")
    projects = []
    for project in workspace.project_list:
        project = Project(api_key=api_key, a_project=project, model_format="coco")
        projects.append(project)
    return DatasetList(projects)


def get_rf20vl_full_projects(api_key: Optional[str] = None) -> DatasetList:
    rf = get_rf(api_key)
    workspace = rf.workspace("rf20-vl")
    projects = []
    for project in workspace.project_list:
        project = Project(api_key=api_key, a_project=project, model_format="coco")
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