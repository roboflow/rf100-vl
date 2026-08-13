"""Fold already-downloaded rf100-vl per-dataset COCO folders into one
combined dataset, in place, without re-fetching or duplicating images.

Two id schemes, per dataset (see `.plans/active/todo_combine-datasets.md`):
- image/annotation/license ids: deterministic offset by the dataset's
  stable global index (`rf100vl.util.get_global_index`), computed
  per-dataset with no cross-dataset coordination.
- category ids: map-reduce over every selected dataset's namespaced
  labels (`f"{basename}:{label}"`), computed once in the join step after
  every dataset's local categories are known.

Destructive by default: source image files are moved (not copied) into
the combined tree, and per-dataset folders are removed once emptied.
Pass `keep_originals=True` to copy instead and leave sources intact.
Progress is tracked in `<path>/.combine_manifest.json` so an interrupted
run can resume without re-reading already-consumed source folders.
"""

import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

from rf100vl.util import get_global_index
from rf100vl.util import values as CANONICAL_BASENAMES

SPLITS = ("train", "valid", "test")
IMAGE_STRIDE = 1_000_000
ANNOTATION_STRIDE = 10_000_000
_INT32_MAX = 2_147_483_647

MANIFEST_FILENAME = ".combine_manifest.json"
_SKIP_DIRS = {".cache", "train", "valid", "test"}


class CombineError(Exception):
    pass


def find_valid_dataset_dirs(path: str) -> List[str]:
    """Return canonical basenames of `path`'s immediate subdirectories that
    look like rf100-vl per-dataset folders (name in the canonical 100-name
    allowlist, at least one split has `_annotations.coco.json`). Non-matching
    dirs are skipped with a warning, not raised.
    """
    canonical = set(CANONICAL_BASENAMES)
    found = []
    for entry in sorted(os.listdir(path)):
        if entry in _SKIP_DIRS or entry.startswith("."):
            continue
        full = os.path.join(path, entry)
        if not os.path.isdir(full):
            continue
        if entry not in canonical:
            print(f"warning: skipping {entry!r} under {path!r} — not a canonical rf100-vl dataset name")
            continue
        if not any(os.path.exists(os.path.join(full, split, "_annotations.coco.json")) for split in SPLITS):
            print(f"warning: skipping {entry!r} under {path!r} — no split has _annotations.coco.json")
            continue
        found.append(entry)
    return found


def _assert_headroom(basename: str, local_ids: List[int], stride: int, global_index: int, label: str) -> None:
    if local_ids and max(local_ids) >= stride:
        raise CombineError(f"{basename}: local {label} id {max(local_ids)} >= stride {stride}")
    if global_index * stride > _INT32_MAX:
        raise CombineError(f"{basename}: global_index {global_index} * {label} stride {stride} overflows int32")


def _remap_split(dataset_dir: str, basename: str, split: str) -> Optional[dict]:
    """Read + remap one dataset's split annotation file. Returns None if the
    dataset has no data for this split. Offsets image/annotation/license ids
    by this dataset's stable global index; namespaces category names
    (`dataset:label`) without assigning a final id yet (decided globally in
    the join step, see `_build_category_map`).
    """
    ann_path = os.path.join(dataset_dir, split, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        return None
    with open(ann_path) as f:
        data = json.load(f)

    global_index = get_global_index(basename)
    image_ids = [img["id"] for img in data["images"]]
    annotation_ids = [ann["id"] for ann in data["annotations"]]
    license_ids = [lic["id"] for lic in data.get("licenses", [])]
    _assert_headroom(basename, image_ids, IMAGE_STRIDE, global_index, "image")
    _assert_headroom(basename, annotation_ids, ANNOTATION_STRIDE, global_index, "annotation")
    _assert_headroom(basename, license_ids, IMAGE_STRIDE, global_index, "license")

    image_offset = global_index * IMAGE_STRIDE
    annotation_offset = global_index * ANNOTATION_STRIDE
    license_offset = global_index * IMAGE_STRIDE

    images = []
    for img in data["images"]:
        new_img = dict(img)
        new_img["id"] = img["id"] + image_offset
        if img.get("license") is not None:
            new_img["license"] = img["license"] + license_offset
        new_img["_source_file_name"] = img["file_name"]
        new_img["file_name"] = f"{basename}_{img['file_name']}"
        images.append(new_img)

    namespaced_by_local_id = {cat["id"]: f"{basename}:{cat['name']}" for cat in data["categories"]}
    annotations = []
    for ann in data["annotations"]:
        new_ann = dict(ann)
        new_ann["id"] = ann["id"] + annotation_offset
        new_ann["image_id"] = ann["image_id"] + image_offset
        new_ann["_namespaced_category"] = namespaced_by_local_id[ann["category_id"]]
        del new_ann["category_id"]
        annotations.append(new_ann)

    licenses = [{**lic, "id": lic["id"] + license_offset} for lic in data.get("licenses", [])]

    return {
        "basename": basename,
        "images": images,
        "annotations": annotations,
        "licenses": licenses,
        "info": data.get("info"),
        "categories": data["categories"],
    }


def _build_category_map(fragments: List[dict]) -> Dict[str, int]:
    """Map-reduce over every selected dataset's namespaced labels: dedupe,
    sort, id = position. Includes categories with zero annotations too.
    """
    labels = set()
    for frag in fragments:
        for cat in frag["categories"]:
            labels.add(f"{frag['basename']}:{cat['name']}")
    return {label: i for i, label in enumerate(sorted(labels))}


def _dedupe_licenses(fragments: List[dict]) -> Tuple[List[dict], Dict[int, int]]:
    """Dedupe already-offset license entries by (url, name) content. Returns
    the deduped license list and a map from offset id -> final id.
    """
    content_to_id: Dict[Tuple, int] = {}
    remap: Dict[int, int] = {}
    deduped: List[dict] = []
    for frag in fragments:
        for lic in frag["licenses"]:
            key = (lic.get("url"), lic.get("name"))
            if key not in content_to_id:
                new_id = len(deduped) + 1
                content_to_id[key] = new_id
                deduped.append({**lic, "id": new_id})
            remap[lic["id"]] = content_to_id[key]
    return deduped, remap


def _merge_split(
    fragments: List[dict],
    category_map: Dict[str, int],
    license_remap: Dict[int, int],
    deduped_licenses: List[dict],
) -> dict:
    categories = []
    for label, cid in sorted(category_map.items(), key=lambda kv: kv[1]):
        supercategory = label.split(":", 1)[0]
        categories.append({"id": cid, "name": label, "supercategory": supercategory})

    images: List[dict] = []
    annotations: List[dict] = []
    infos: List[dict] = []
    for frag in fragments:
        for img in frag["images"]:
            new_img = {k: v for k, v in img.items() if not k.startswith("_")}
            if new_img.get("license") is not None:
                new_img["license"] = license_remap[new_img["license"]]
            images.append(new_img)
        for ann in frag["annotations"]:
            new_ann = {k: v for k, v in ann.items() if not k.startswith("_")}
            new_ann["category_id"] = category_map[ann["_namespaced_category"]]
            annotations.append(new_ann)
        if frag["info"]:
            infos.append(frag["info"])

    info: dict = {}
    if infos:
        info = dict(infos[0]) if isinstance(infos[0], dict) else {}
        info["rf100vl_source_info"] = infos

    return {
        "info": info,
        "licenses": deduped_licenses,
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def _load_manifest(path: str) -> dict:
    manifest_path = os.path.join(path, MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {"datasets": {}}


def _save_manifest(path: str, manifest: dict) -> None:
    manifest_path = os.path.join(path, MANIFEST_FILENAME)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp_path, manifest_path)


def _materialize_dataset(
    path: str, out_path: str, basename: str, fragments: Dict[str, dict], keep_originals: bool
) -> None:
    """Move (or copy) one dataset's images from `path/<basename>` into the
    combined tree at `out_path`. Idempotent per-file — safe to re-run after
    a partial interruption: a file already present at its destination is
    left alone, and a file missing from both source and destination raises
    loudly instead of silently producing a combined dataset with missing
    images. `keep_originals` implies copy — always true when `path !=
    out_path` (Flow A: source lives under `.cache`, must survive for reuse).
    """
    dataset_dir = os.path.join(path, basename)
    keep_originals = keep_originals or path != out_path
    for split, frag in fragments.items():
        images_dir = os.path.join(out_path, split, "images")
        os.makedirs(images_dir, exist_ok=True)
        for img in frag["images"]:
            src = os.path.join(dataset_dir, split, img["_source_file_name"])
            dst = os.path.join(images_dir, img["file_name"])
            if os.path.exists(dst):
                continue
            if not os.path.exists(src):
                raise CombineError(f"{basename}/{split}: missing both source {src!r} and dest {dst!r}")
            if keep_originals:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
        if not keep_originals:
            ann_path = os.path.join(dataset_dir, split, "_annotations.coco.json")
            if os.path.exists(ann_path):
                os.remove(ann_path)

    if not keep_originals:
        for split in SPLITS:
            split_dir = os.path.join(dataset_dir, split)
            if os.path.isdir(split_dir) and not os.listdir(split_dir):
                os.rmdir(split_dir)
        # Roboflow's COCO export always drops these two boilerplate files at
        # the dataset root (confirmed identical structure across datasets);
        # safe to remove by name. Anything else unexpected left over still
        # blocks the final rmdir below rather than being force-deleted.
        for readme in ("README.roboflow.txt", "README.dataset.txt"):
            readme_path = os.path.join(dataset_dir, readme)
            if os.path.exists(readme_path):
                os.remove(readme_path)
        if os.path.isdir(dataset_dir) and not os.listdir(dataset_dir):
            os.rmdir(dataset_dir)


def combine(
    path: str,
    basenames: Optional[List[str]] = None,
    keep_originals: bool = False,
    out_path: Optional[str] = None,
) -> Dict[str, dict]:
    """Combine per-dataset folders found under `path` into one dataset at
    `out_path/{train,valid,test}`. Returns the combined per-split COCO
    dicts (also written to disk).

    `basenames`: which per-dataset folders to include; default is every
    valid one found directly under `path` (see `find_valid_dataset_dirs`).
    `out_path`: defaults to `path` (Flow B: true in-place combine). Pass a
    different `out_path` to read sources from one directory (e.g. a
    `.cache/` of raw downloads) while writing the combined tree elsewhere
    (Flow A) — implies `keep_originals` so the cache survives for reuse.
    """
    out_path = out_path or path
    os.makedirs(out_path, exist_ok=True)
    selected = basenames if basenames is not None else find_valid_dataset_dirs(path)
    if not selected:
        raise CombineError(f"no valid rf100-vl dataset folders found under {path!r}")
    invalid = sorted(set(selected) - set(CANONICAL_BASENAMES))
    if invalid:
        raise CombineError(f"not canonical rf100-vl dataset names: {invalid}")

    manifest = _load_manifest(out_path)
    datasets_cache = manifest.setdefault("datasets", {})

    for basename in selected:
        entry = datasets_cache.get(basename)
        if entry is not None and entry.get("materialized"):
            continue

        if entry is None:
            dataset_dir = os.path.join(path, basename)
            if not os.path.isdir(dataset_dir):
                raise CombineError(f"{basename}: no such directory {dataset_dir!r}")
            fragments = {}
            for split in SPLITS:
                frag = _remap_split(dataset_dir, basename, split)
                if frag is not None:
                    fragments[split] = frag
            if not fragments:
                raise CombineError(f"{basename}: no split has _annotations.coco.json under {dataset_dir!r}")
            entry = {"fragments": fragments, "materialized": False}
            datasets_cache[basename] = entry
            _save_manifest(out_path, manifest)

        _materialize_dataset(path, out_path, basename, entry["fragments"], keep_originals)
        entry["materialized"] = True
        _save_manifest(out_path, manifest)

    # Join over every materialized dataset in the manifest, not just
    # `selected` — a prior run's dataset may no longer exist as a directory
    # (already moved+removed) so it wouldn't be rediscovered by a default
    # directory scan, but its images already live under path/{split}/images/
    # and must stay referenced in the combined json or they'd be orphaned.
    combined_basenames = sorted(b for b, entry in datasets_cache.items() if entry.get("materialized"))
    all_fragments = [
        datasets_cache[b]["fragments"][s] for b in combined_basenames for s in SPLITS if s in datasets_cache[b]["fragments"]
    ]
    category_map = _build_category_map(all_fragments)
    deduped_licenses, license_remap = _dedupe_licenses(all_fragments)

    combined_by_split: Dict[str, dict] = {}
    for split in SPLITS:
        split_fragments = [
            datasets_cache[b]["fragments"][split] for b in combined_basenames if split in datasets_cache[b]["fragments"]
        ]
        if not split_fragments:
            continue
        combined = _merge_split(split_fragments, category_map, license_remap, deduped_licenses)
        combined_by_split[split] = combined
        out_dir = os.path.join(out_path, split)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
            json.dump(combined, f)

    return combined_by_split
