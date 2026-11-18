from types import SimpleNamespace

from rf100vl.dataset import RF100VlDataset


def _dataset(project_name: str) -> RF100VlDataset:
    return RF100VlDataset(SimpleNamespace(name=project_name))


def test_is_fewshot_for_fsod_project():
    dataset = _dataset("13-lkc01-fsod-vjnp")

    assert dataset.name == "13-lkc01"
    assert dataset.is_fewshot is True


def test_is_fewshot_for_full_project():
    dataset = _dataset("13-lkc01-axmll-qtoq")

    assert dataset.name == "13-lkc01"
    assert dataset.is_fewshot is False
