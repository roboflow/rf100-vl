import os
import json
from typing import Dict, List, Optional, Iterator
from rf100vl.dataset import RF100VlDataset
from rf100vl.combine import combine as run_combine
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


# ---------------------------------------------------------------------------
# Combine — fold per-dataset COCO folders into one dataset. See
# rf100vl/combine.py for the actual remap/merge/move logic and
# .plans/active/todo_combine-datasets.md for the design.
# ---------------------------------------------------------------------------


def combine_downloaded(path: str, basenames: Optional[List[str]] = None, keep_originals: bool = False) -> Dict[str, dict]:
    """Fold per-dataset folders already downloaded under ``path`` into one
    combined COCO dataset directly at ``path/{train,valid,test}`` — no
    network access. Thin wrapper over ``rf100vl.combine.combine`` (Flow B,
    in-place). Destructive by default: source folders are moved and
    removed once empty; pass ``keep_originals=True`` to copy instead.

    ``basenames``: which per-dataset folders to include; default is every
    valid one found directly under ``path``.
    """
    return run_combine(path, basenames=basenames, keep_originals=keep_originals)


def _resolve_datasets_for_combine(
    indices: Optional[List[int]],
    names: Optional[List[str]],
    variant: str,
    api_key: Optional[str],
) -> List[RF100VlDataset]:
    if (indices is None) == (names is None):
        raise ValueError("pass exactly one of `indices` or `names`")
    fetcher = _PROJECT_FETCHERS.get(variant)
    if fetcher is None:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_PROJECT_FETCHERS)}")
    all_datasets = fetcher(api_key)
    if indices is not None:
        n = len(all_datasets)
        for i in indices:
            if not 0 <= i < n:
                raise IndexError(f"variant {variant!r} has {n} datasets; index {i} out of range")
        return [all_datasets[i] for i in indices]
    by_name = {d.name: d for d in all_datasets}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"unknown dataset name(s) for variant {variant!r}: {missing}")
    return [by_name[n] for n in names]


def _load_combine_manifest(path: str) -> dict:
    manifest_path = os.path.join(path, ".combine_manifest.json")
    if not os.path.exists(manifest_path):
        return {"datasets": {}}
    with open(manifest_path) as f:
        return json.load(f)


def _save_combine_manifest(path: str, manifest: dict) -> None:
    os.makedirs(path, exist_ok=True)
    manifest_path = os.path.join(path, ".combine_manifest.json")
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp_path, manifest_path)


def _dataset_cache_complete(cache_dir: str, basename: str) -> bool:
    dataset_dir = os.path.join(cache_dir, basename)
    if not os.path.isdir(dataset_dir):
        return False
    return any(
        os.path.exists(os.path.join(dataset_dir, split, "_annotations.coco.json"))
        for split in ("train", "valid", "test")
    )


def _invalidate_materialized_dataset(path: str, manifest: dict, basename: str) -> None:
    datasets_cache = manifest.get("datasets", {})
    entry = datasets_cache.get(basename)
    if not entry or not entry.get("materialized"):
        return
    for split, frag in entry.get("fragments", {}).items():
        images_dir = os.path.join(path, split, "images")
        for img in frag.get("images", []):
            image_path = os.path.join(images_dir, img["file_name"])
            if os.path.exists(image_path):
                os.remove(image_path)
    del datasets_cache[basename]


def download_and_combine(
    path: str,
    indices: Optional[List[int]] = None,
    names: Optional[List[str]] = None,
    *,
    variant: str = "rf100vl",
    model_format: str = "coco",
    overwrite: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, dict]:
    """Download the selected dataset(s) and fold them into one combined COCO
    tree directly at ``path/{train,valid,test}``. Raw per-dataset downloads
    land under ``path/.cache/`` and are kept (not cleaned up) so a later
    call with an overlapping selection can reuse them (Flow A).

    Pass exactly one of ``indices`` (global indices into the sorted
    variant) or ``names`` (canonical basenames).
    """
    if model_format != "coco":
        raise ValueError(f"combine only supports model_format='coco', got {model_format!r}")
    datasets = _resolve_datasets_for_combine(indices, names, variant, api_key)
    cache_dir = os.path.join(path, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    manifest = _load_combine_manifest(path)
    datasets_cache = manifest.get("datasets", {})
    manifest_dirty = False
    for dataset in datasets:
        entry = datasets_cache.get(dataset.name)
        materialized = bool(entry and entry.get("materialized"))
        if overwrite and materialized:
            _invalidate_materialized_dataset(path, manifest, dataset.name)
            entry = None
            materialized = False
            manifest_dirty = True
        cache_complete = _dataset_cache_complete(cache_dir, dataset.name)
        should_download = overwrite or not cache_complete
        if materialized and not overwrite:
            should_download = False
        if should_download:
            dataset.download(os.path.join(cache_dir, dataset.name), model_format=model_format, overwrite=overwrite)
    if manifest_dirty:
        _save_combine_manifest(path, manifest)
    basenames = [dataset.name for dataset in datasets]
    return run_combine(cache_dir, basenames=basenames, out_path=path)