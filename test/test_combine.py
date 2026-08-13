"""Tests for rf100vl.combine — pure remap/merge helpers and the destructive
end-to-end combine() flow, run against throwaway copies of the two smallest
local datasets (jellyfish, deeppcb) so a bug can never touch the real
40GB corpus.
"""

import json
import os
import shutil

import pytest

from rf100vl import __main__ as cli
from rf100vl import combine as combine_mod
from rf100vl import roboflow100vl
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
        assert isinstance(result["train"]["info"], dict)
        assert "rf100vl_source_info" in result["train"]["info"]

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

    def test_raises_on_explicit_basename_with_no_directory(self, tmp_path):
        """A canonical basename with no matching directory under path must
        raise, not silently return an empty result with no error -- an
        earlier version of this code did exactly that.
        """
        with pytest.raises(CombineError):
            combine(str(tmp_path), basenames=["bees"])

    @requires_local_corpus
    def test_out_path_is_created_before_manifest_write(self, corpus_copy):
        out_path = corpus_copy / "combined"

        combine(str(corpus_copy), basenames=["jellyfish"], out_path=str(out_path))

        assert os.path.exists(out_path / ".combine_manifest.json")


class TestCombineDownloaded:
    @requires_local_corpus
    def test_thin_wrapper_over_combine(self, corpus_copy):
        """combine_downloaded delegates to combine() with the same semantics."""
        result = roboflow100vl.combine_downloaded(str(corpus_copy))

        assert not os.path.exists(corpus_copy / "jellyfish")
        assert len(result["train"]["images"]) > 0


class _FakeDataset:
    """Stands in for RF100VlDataset in download_and_combine wiring tests --
    `.download()` copies from a local fixture instead of hitting the network.
    """

    def __init__(self, name, source_dir):
        self.name = name
        self._source_dir = source_dir

    def download(self, path, model_format="coco", overwrite=True):
        shutil.copytree(self._source_dir, path, dirs_exist_ok=True)


class _CountingFakeDataset(_FakeDataset):
    def __init__(self, name, source_dir):
        super().__init__(name, source_dir)
        self.download_calls = 0

    def download(self, path, model_format="coco", overwrite=True):
        self.download_calls += 1
        super().download(path, model_format=model_format, overwrite=overwrite)


class TestResolveDatasetsForCombine:
    def test_requires_exactly_one_of_indices_or_names(self):
        """Passing both or neither indices/names is a hard error, not an ambiguous default."""
        with pytest.raises(ValueError):
            roboflow100vl._resolve_datasets_for_combine(None, None, "rf100vl", None)

    def test_unknown_variant_raises(self):
        """An unsupported variant name fails loudly instead of silently no-oping."""
        with pytest.raises(ValueError):
            roboflow100vl._resolve_datasets_for_combine([0], None, "not-a-real-variant", None)

    def test_indices_validate_non_negative(self, monkeypatch):
        monkeypatch.setitem(roboflow100vl._PROJECT_FETCHERS, "rf100vl", lambda api_key=None: [_FakeDataset("bees", "")])

        with pytest.raises(IndexError):
            roboflow100vl._resolve_datasets_for_combine([-1], None, "rf100vl", None)


class TestDownloadAndCombine:
    @requires_local_corpus
    def test_downloads_into_cache_and_combines_at_top_level(self, tmp_path, monkeypatch):
        """Raw downloads land under path/.cache/<name>/ and survive (Flow A
        always keeps originals); combined tree is written at path/ directly.
        No real network call -- RF100VlDataset.download is faked to copy
        from the local jellyfish/deeppcb fixtures.
        """
        fakes = [
            _FakeDataset("jellyfish", os.path.join(_LOCAL_CORPUS, "jellyfish")),
            _FakeDataset("deeppcb", os.path.join(_LOCAL_CORPUS, "deeppcb")),
        ]
        monkeypatch.setitem(roboflow100vl._PROJECT_FETCHERS, "rf100vl", lambda api_key=None: fakes)

        result = roboflow100vl.download_and_combine(str(tmp_path), names=["jellyfish", "deeppcb"])

        assert os.path.exists(tmp_path / ".cache" / "jellyfish" / "train" / "_annotations.coco.json")
        assert os.path.exists(tmp_path / ".cache" / "deeppcb" / "train" / "_annotations.coco.json")
        assert os.path.exists(tmp_path / "train" / "_annotations.coco.json")
        assert len(result["train"]["images"]) > 0

    @requires_local_corpus
    def test_reuses_materialized_entry_without_redownload(self, tmp_path, monkeypatch):
        fake = _CountingFakeDataset("jellyfish", os.path.join(_LOCAL_CORPUS, "jellyfish"))
        monkeypatch.setitem(roboflow100vl._PROJECT_FETCHERS, "rf100vl", lambda api_key=None: [fake])

        roboflow100vl.download_and_combine(str(tmp_path), names=["jellyfish"], overwrite=False)
        roboflow100vl.download_and_combine(str(tmp_path), names=["jellyfish"], overwrite=False)

        assert fake.download_calls == 1


class TestSplitCsvFlag:
    """Fire hands back different Python types depending on the flag value's
    shape, not always the raw string -- a bug (`'tuple' object has no
    attribute 'split'`) shipped because tests called cli functions
    directly with plain strings and never exercised this.
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            pytest.param(None, None, id="none"),
            pytest.param("0,15,29", ["0", "15", "29"], id="str-multi"),
            pytest.param("5", ["5"], id="str-single"),
            pytest.param((0, 15, 29), ["0", "15", "29"], id="fire-tuple-multi"),
            pytest.param(5, ["5"], id="fire-bare-int-single"),
        ],
    )
    def test_normalizes_every_shape_fire_can_produce(self, value, expected):
        assert cli._split_csv_flag(value) == expected


class TestCLICombine:
    @requires_local_corpus
    def test_names_resolve_and_combine(self, corpus_copy):
        """`rf100vl combine --names` resolves directly to basenames."""
        cli.combine(str(corpus_copy), names="jellyfish,deeppcb")

        assert not os.path.exists(corpus_copy / "jellyfish")
        assert os.path.exists(corpus_copy / "train" / "_annotations.coco.json")

    @requires_local_corpus
    def test_indices_resolve_to_basenames(self, corpus_copy):
        """`rf100vl combine --indices` resolves global indices to basenames
        via the same canonical list used by get_global_index."""
        jelly_idx = get_global_index("jellyfish")
        pcb_idx = get_global_index("deeppcb")

        cli.combine(str(corpus_copy), indices=f"{jelly_idx},{pcb_idx}")

        assert not os.path.exists(corpus_copy / "jellyfish")
        assert not os.path.exists(corpus_copy / "deeppcb")

    @requires_local_corpus
    def test_indices_as_fire_parsed_tuple(self, corpus_copy):
        """Regression test for the actual reported crash: Fire parses
        `--indices 0,15,29` into a tuple of ints before combine() ever
        sees it, not the comma-separated string `.split(",")` assumed.
        """
        jelly_idx = get_global_index("jellyfish")
        pcb_idx = get_global_index("deeppcb")

        cli.combine(str(corpus_copy), indices=(jelly_idx, pcb_idx))

        assert not os.path.exists(corpus_copy / "jellyfish")
        assert not os.path.exists(corpus_copy / "deeppcb")

    def test_indices_and_names_mutually_exclusive(self, tmp_path):
        """Passing both --indices and --names is a hard error."""
        with pytest.raises(ValueError):
            cli.combine(str(tmp_path), indices="0", names="bees")

    def test_negative_indices_are_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            cli.combine(str(tmp_path), indices="-1")


class TestCLIDownloadCombine:
    def test_combine_requires_a_selection(self, tmp_path):
        """--combine with no --index/--indices/--names is a hard error, not
        an implicit "combine all 100 datasets"."""
        with pytest.raises(ValueError):
            cli.download("rf100vl", str(tmp_path), combine=True)

    def test_combine_rejects_fsod(self, tmp_path):
        """--combine + --fsod is unsupported for now, fails loudly."""
        with pytest.raises(ValueError):
            cli.download("rf100vl", str(tmp_path), fsod=True, combine=True, index=0)

    def test_combine_indices_and_names_mutually_exclusive(self, tmp_path):
        """Passing both --indices and --names with --combine is a hard error."""
        with pytest.raises(ValueError):
            cli.download("rf100vl", str(tmp_path), combine=True, indices="0", names="bees")

    def test_indices_require_combine(self, tmp_path):
        with pytest.raises(ValueError):
            cli.download("rf100vl", str(tmp_path), indices="0")

    def test_names_require_combine(self, tmp_path):
        with pytest.raises(ValueError):
            cli.download("rf100vl", str(tmp_path), names="bees")

    def test_combine_selection_is_strictly_mutually_exclusive(self, tmp_path):
        with pytest.raises(ValueError):
            cli.download("rf100vl", str(tmp_path), combine=True, index=0, indices="0")
        with pytest.raises(ValueError):
            cli.download("rf100vl", str(tmp_path), combine=True, index=0, names="bees")

    @requires_local_corpus
    def test_combine_wires_through_to_download_and_combine(self, tmp_path, monkeypatch):
        """`rf100vl download --combine --names=...` downloads (faked, no
        network) into path/.cache/ and writes the combined tree at path/.
        """
        fakes = [
            _FakeDataset("jellyfish", os.path.join(_LOCAL_CORPUS, "jellyfish")),
            _FakeDataset("deeppcb", os.path.join(_LOCAL_CORPUS, "deeppcb")),
        ]
        monkeypatch.setitem(roboflow100vl._PROJECT_FETCHERS, "rf100vl", lambda api_key=None: fakes)

        cli.download("rf100vl", str(tmp_path), combine=True, names="jellyfish,deeppcb")

        assert os.path.exists(tmp_path / ".cache" / "jellyfish" / "train" / "_annotations.coco.json")
        assert os.path.exists(tmp_path / "train" / "_annotations.coco.json")

    @requires_local_corpus
    def test_combine_indices_as_fire_parsed_tuple(self, tmp_path, monkeypatch):
        """Same regression as TestCLICombine.test_indices_as_fire_parsed_tuple,
        for the download --combine --indices path."""
        fakes = [
            _FakeDataset("jellyfish", os.path.join(_LOCAL_CORPUS, "jellyfish")),
            _FakeDataset("deeppcb", os.path.join(_LOCAL_CORPUS, "deeppcb")),
        ]
        monkeypatch.setitem(roboflow100vl._PROJECT_FETCHERS, "rf100vl", lambda api_key=None: fakes)

        cli.download("rf100vl", str(tmp_path), combine=True, indices=(0, 1))

        assert os.path.exists(tmp_path / "train" / "_annotations.coco.json")
