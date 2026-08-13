"""Tests for rf100vl.combine — pure remap/merge helpers and the destructive
end-to-end combine() flow, run against throwaway copies of the two smallest
local datasets (jellyfish, deeppcb) so a bug can never touch the real
40GB corpus.
"""

import json
import os
import shutil

import pytest

from rf100vl import combine as combine_mod
from rf100vl.combine import CombineError, combine, find_valid_dataset_dirs
from rf100vl.util import get_global_index

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_CORPUS = os.path.join(_REPO_ROOT, "rf100-vl")
_FIXTURE_DATASETS = ("jellyfish", "deeppcb")


def _fixtures_available() -> bool:
    return all(
        os.path.exists(os.path.join(_LOCAL_CORPUS, name, "train", "_annotations.coco.json"))
        for name in _FIXTURE_DATASETS
    )


requires_local_corpus = pytest.mark.skipif(
    not _fixtures_available(), reason="local rf100-vl corpus not present (jellyfish/deeppcb)"
)


@pytest.fixture
def corpus_copy(tmp_path):
    """Copy jellyfish + deeppcb into an isolated tmp dir so destructive
    combine() runs never touch the real local corpus.
    """
    for name in _FIXTURE_DATASETS:
        shutil.copytree(os.path.join(_LOCAL_CORPUS, name), tmp_path / name)
    return tmp_path


class TestFindValidDatasetDirs:
    @requires_local_corpus
    def test_finds_canonical_dataset_dirs(self, corpus_copy):
        """Both copied fixture datasets are recognized as valid."""
        found = find_valid_dataset_dirs(str(corpus_copy))

        assert sorted(found) == sorted(_FIXTURE_DATASETS)

    def test_skips_non_canonical_dir(self, tmp_path):
        """A dir whose name isn't one of the 100 canonical basenames is skipped."""
        os.makedirs(tmp_path / "not-a-real-dataset" / "train")

        found = find_valid_dataset_dirs(str(tmp_path))

        assert found == []

    def test_skips_own_output_dirs(self, tmp_path):
        """`.cache`, `train`, `valid`, `test` are always skipped, even if present."""
        for name in (".cache", "train", "valid", "test"):
            os.makedirs(tmp_path / name)

        found = find_valid_dataset_dirs(str(tmp_path))

        assert found == []


class TestBuildCategoryMap:
    def test_namespaces_and_dedupes_labels(self):
        """Same label string in two datasets stays two distinct namespaced ids."""
        fragments = [
            {"basename": "deeppcb", "categories": [{"name": "open"}, {"name": "short"}]},
            {"basename": "stomata-cells", "categories": [{"name": "open"}]},
        ]

        category_map = combine_mod._build_category_map(fragments)

        assert set(category_map) == {"deeppcb:open", "deeppcb:short", "stomata-cells:open"}
        assert category_map["deeppcb:open"] != category_map["stomata-cells:open"]

    def test_ids_are_sorted_position(self):
        """global_category_id is the position of the label in sorted order."""
        fragments = [{"basename": "z", "categories": [{"name": "b"}]}, {"basename": "a", "categories": [{"name": "b"}]}]

        category_map = combine_mod._build_category_map(fragments)

        assert category_map == {"a:b": 0, "z:b": 1}


class TestAssertHeadroom:
    @pytest.mark.parametrize(
        "local_ids,stride,global_index",
        [
            pytest.param([0, 1, 5], 5, 0, id="local-id-exceeds-stride"),
            pytest.param([0], 1_000_000, 3000, id="global-index-overflows-int32"),
        ],
    )
    def test_raises_on_overflow(self, local_ids, stride, global_index):
        """Both overflow conditions (local id headroom, int32 ceiling) raise, not silently collide."""
        with pytest.raises(CombineError):
            combine_mod._assert_headroom("ds", local_ids, stride, global_index, "image")

    def test_passes_within_headroom(self):
        """Real observed max (8791 images) stays under IMAGE_STRIDE with room to spare."""
        combine_mod._assert_headroom("bees", [8791], combine_mod.IMAGE_STRIDE, get_global_index("bees"), "image")


class TestCombineEndToEnd:
    @requires_local_corpus
    def test_combines_two_datasets(self, corpus_copy):
        """Images move into path/{split}/images/, source dirs are removed,
        combined json has correct totals and namespaced categories."""
        jelly_json = json.load(open(corpus_copy / "jellyfish" / "train" / "_annotations.coco.json"))
        pcb_json = json.load(open(corpus_copy / "deeppcb" / "train" / "_annotations.coco.json"))
        expected_images = len(jelly_json["images"]) + len(pcb_json["images"])
        expected_anns = len(jelly_json["annotations"]) + len(pcb_json["annotations"])

        result = combine(str(corpus_copy))

        assert len(result["train"]["images"]) == expected_images
        assert len(result["train"]["annotations"]) == expected_anns
        assert set(c["name"] for c in result["train"]["categories"]) == {
            "jellyfish:Jellyfish",
            "deeppcb:copper",
            "deeppcb:mousebite",
            "deeppcb:open",
            "deeppcb:pin-hole",
            "deeppcb:short",
            "deeppcb:spur",
        }
        assert not os.path.exists(corpus_copy / "jellyfish")
        assert not os.path.exists(corpus_copy / "deeppcb")
        assert os.path.exists(corpus_copy / "train" / "images")
        moved_files = os.listdir(corpus_copy / "train" / "images")
        assert any(f.startswith("jellyfish_") for f in moved_files)
        assert any(f.startswith("deeppcb_") for f in moved_files)

    @requires_local_corpus
    def test_keep_originals_copies_instead_of_moves(self, corpus_copy):
        """keep_originals=True leaves source per-dataset folders intact."""
        combine(str(corpus_copy), keep_originals=True)

        assert os.path.exists(corpus_copy / "jellyfish" / "train" / "_annotations.coco.json")
        assert os.path.exists(corpus_copy / "deeppcb" / "train" / "_annotations.coco.json")
        assert os.path.exists(corpus_copy / "train" / "images")

    @requires_local_corpus
    def test_resume_skips_already_materialized_dataset(self, corpus_copy):
        """A dataset already combined in a prior run is not reprocessed --
        the manifest lets a second call add the remaining dataset only."""
        combine(str(corpus_copy), basenames=["jellyfish"])
        assert not os.path.exists(corpus_copy / "jellyfish")
        assert os.path.exists(corpus_copy / "deeppcb")

        result = combine(str(corpus_copy))

        assert not os.path.exists(corpus_copy / "deeppcb")
        assert any(name.startswith("jellyfish:") for name in [c["name"] for c in result["train"]["categories"]])
        assert any(name.startswith("deeppcb:") for name in [c["name"] for c in result["train"]["categories"]])

    def test_raises_on_no_valid_datasets(self, tmp_path):
        """Empty/non-matching path is a hard error, not a silent no-op."""
        with pytest.raises(CombineError):
            combine(str(tmp_path))

    def test_raises_on_non_canonical_basename(self, tmp_path):
        """Explicitly passing a non-canonical name is a hard error."""
        os.makedirs(tmp_path / "totally-not-real")
        with pytest.raises(CombineError):
            combine(str(tmp_path), basenames=["totally-not-real"])
