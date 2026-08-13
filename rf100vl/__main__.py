from typing import Optional

from rf100vl import (
    combine_downloaded,
    download_and_combine,
    download_rf100vl,
    download_rf100vl_fsod,
    download_rf100vl_index,
    download_rf20vl_fsod,
    download_rf20vl_full,
)
from rf100vl.util import values as _CANONICAL_BASENAMES

_FULL_DOWNLOADERS = {
    "rf100vl": download_rf100vl,
    "rf20vl": download_rf20vl_full,
}
_FSOD_DOWNLOADERS = {
    "rf100vl": download_rf100vl_fsod,
    "rf20vl": download_rf20vl_fsod,
}
_INDEX_VARIANTS = {
    ("rf100vl", False): "rf100vl",
    ("rf100vl", True): "rf100vl_fsod",
    ("rf20vl", False): "rf20vl_full",
    ("rf20vl", True): "rf20vl_fsod",
}


def _indices_to_basenames(indices: list) -> list:
    basenames = []
    n = len(_CANONICAL_BASENAMES)
    for token in indices:
        i = int(token)
        if not 0 <= i < n:
            raise ValueError(f"index {i} out of range (expected 0..{n - 1})")
        basenames.append(_CANONICAL_BASENAMES[i])
    return basenames


def _split_csv_flag(value) -> Optional[list]:
    """Normalize a comma-separated CLI flag to a list of str tokens.

    Fire doesn't hand back the raw string: a multi-value flag like
    `--indices 0,15,29` is parsed into a tuple `(0, 15, 29)` before this
    function ever sees it, and a single value like `--indices 5` is parsed
    into a bare `int`. Handle every shape Fire can produce, not just str.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [v.strip() for v in str(value).split(",")]


def download(
    dataset: str,
    path: str,
    fsod: bool = False,
    index: Optional[int] = None,
    indices: Optional[str] = None,
    names: Optional[str] = None,
    model_format: str = "coco",
    overwrite: bool = True,
    api_key: Optional[str] = None,
    combine: bool = False,
) -> None:
    """Download an rf100-vl dataset.

    dataset: "rf100vl" or "rf20vl"
    fsod: use the few-shot object detection variant
    index: download only the dataset at this zero-based index in the variant, instead of all
    indices: comma-separated global indices to download and fold together, e.g. "0,3,7";
        only valid together with combine=True, mutually exclusive with names
    names: comma-separated canonical dataset basenames to download and fold together, e.g.
        "bees,deeppcb"; only valid together with combine=True, mutually exclusive with indices
    combine: also fold the downloaded dataset(s) into a combined tree at
        `<path>/{train,valid,test}` after downloading; raw per-dataset downloads are kept
        under `<path>/.cache/` (reused on future combine calls, not cleaned up automatically).
        Requires index, indices, or names to pick a finite selection; fsod not supported yet.
    """
    if dataset not in _FULL_DOWNLOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(_FULL_DOWNLOADERS)}")

    if not combine and (indices is not None or names is not None):
        raise ValueError("--indices and --names are only valid with --combine")

    if combine:
        if fsod:
            raise ValueError("combine is not supported for the fsod variant yet")
        split_indices = _split_csv_flag(indices)
        parsed_indices = [int(i) for i in split_indices] if split_indices is not None else None
        parsed_names = _split_csv_flag(names)
        selection_count = int(index is not None) + int(parsed_indices is not None) + int(parsed_names is not None)
        if selection_count != 1:
            raise ValueError("combine requires exactly one of --index, --indices, or --names")
        if index is not None:
            parsed_indices = [index]
        variant = _INDEX_VARIANTS[(dataset, False)]
        download_and_combine(
            path,
            indices=parsed_indices,
            names=parsed_names,
            variant=variant,
            model_format=model_format,
            overwrite=overwrite,
            api_key=api_key,
        )
        return

    if index is not None:
        variant = _INDEX_VARIANTS[(dataset, fsod)]
        download_rf100vl_index(
            index, path, variant=variant, model_format=model_format, overwrite=overwrite, api_key=api_key
        )
        return

    downloader = (_FSOD_DOWNLOADERS if fsod else _FULL_DOWNLOADERS)[dataset]
    downloader(path, model_format=model_format, overwrite=overwrite, api_key=api_key)


def combine(
    path: str,
    indices: Optional[str] = None,
    names: Optional[str] = None,
    keep_originals: bool = False,
) -> None:
    """Fold rf100-vl subdatasets already downloaded under `path` into one
    combined COCO dataset directly at `path/{train,valid,test}`.

    Validates each immediate subdirectory of `path` against the canonical
    100 rf100-vl dataset names before including it; non-matching or
    malformed dirs are skipped with a warning. No network access.

    Categories are deduped and dataset-namespaced (`dataset:label`);
    image/annotation ids are offset per dataset via a stable global index,
    not selection order, so re-running with a different subset doesn't
    reshuffle existing ids.

    DESTRUCTIVE BY DEFAULT: per-dataset source folders (e.g. `path/bees`)
    are moved into the combined tree and removed once empty — not copied.
    Pass keep_originals=True to copy instead and leave source folders
    intact. Interruptible and resumable: progress is tracked in
    `path/.combine_manifest.json`; re-running skips already-merged datasets.

    path: directory containing the per-dataset folders to combine
    indices: comma-separated global dataset indices to include, e.g. "0,3,7"; mutually
        exclusive with names; default is every valid dataset found directly under path
    names: comma-separated canonical dataset basenames to include, e.g. "bees,deeppcb";
        mutually exclusive with indices
    keep_originals: copy/hardlink instead of move — leaves source per-dataset folders
        untouched (default: False, move+remove)
    """
    if indices is not None and names is not None:
        raise ValueError("pass at most one of --indices or --names")
    basenames = None
    split_indices = _split_csv_flag(indices)
    if split_indices is not None:
        basenames = _indices_to_basenames(split_indices)
    else:
        basenames = _split_csv_flag(names)
    combine_downloaded(path, basenames=basenames, keep_originals=keep_originals)


def main() -> None:
    try:
        import fire
    except ImportError as e:
        raise ImportError('CLI requires the "cli" extra: pip install "rf100vl[cli]"') from e
    fire.Fire({"download": download, "combine": combine})


if __name__ == "__main__":
    main()
