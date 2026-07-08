from typing import Optional

from rf100vl import (
    download_rf100vl,
    download_rf100vl_fsod,
    download_rf100vl_index,
    download_rf20vl_fsod,
    download_rf20vl_full,
)

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


def download(
    dataset: str,
    path: str,
    fsod: bool = False,
    index: Optional[int] = None,
    model_format: str = "coco",
    overwrite: bool = True,
    api_key: Optional[str] = None,
) -> None:
    """Download an rf100-vl dataset.

    dataset: "rf100vl" or "rf20vl"
    fsod: use the few-shot object detection variant
    index: download only the dataset at this zero-based index in the variant, instead of all
    """
    if dataset not in _FULL_DOWNLOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(_FULL_DOWNLOADERS)}")

    if index is not None:
        variant = _INDEX_VARIANTS[(dataset, fsod)]
        download_rf100vl_index(
            index, path, variant=variant, model_format=model_format, overwrite=overwrite, api_key=api_key
        )
        return

    downloader = (_FSOD_DOWNLOADERS if fsod else _FULL_DOWNLOADERS)[dataset]
    downloader(path, model_format=model_format, overwrite=overwrite, api_key=api_key)


def main() -> None:
    try:
        import fire
    except ImportError as e:
        raise ImportError('CLI requires the "cli" extra: pip install "rf100vl[cli]"') from e
    fire.Fire({"download": download})


if __name__ == "__main__":
    main()
