import json
import os

DATASET_TO_CATEGORY_JSON_PATH = os.path.join(os.path.dirname(__file__), "assets", "dataset_name_to_category.json")
DATASET_TO_BASENAME_JSON_PATH = os.path.join(os.path.dirname(__file__), "assets", "dataset_name_to_basename.json")

def get_category_json():
    with open(DATASET_TO_CATEGORY_JSON_PATH, "r") as f:
        return json.load(f)
    
def get_basename_json():
    with open(DATASET_TO_BASENAME_JSON_PATH, "r") as f:
        dataset_to_basename = json.load(f)
    
    assert len(set(dataset_to_basename.values())) == 100
    return dataset_to_basename

DATASET_TO_CATEGORY_JSON = get_category_json()
DATASET_TO_BASENAME_JSON = get_basename_json()
keys = list(sorted(set(DATASET_TO_CATEGORY_JSON.keys())))
values = list(sorted(set(DATASET_TO_BASENAME_JSON.values())))
assert set(DATASET_TO_CATEGORY_JSON.keys()) == set(DATASET_TO_BASENAME_JSON.values())
assert len(keys) == len(values) == 100

def get_category(dataset_name: str):
    return DATASET_TO_CATEGORY_JSON[dataset_name]

def get_basename(dataset_name: str):
    return DATASET_TO_BASENAME_JSON[dataset_name]

# Stable global rank per canonical basename (0..99), independent of selection
# order, download state, or DatasetList's sort. Derived from `values` (already
# the sorted canonical basename list above) rather than a new asset file, so
# there is one source of truth for the 100-dataset name list.
BASENAME_TO_GLOBAL_INDEX = {name: i for i, name in enumerate(values)}

def get_global_index(basename: str) -> int:
    return BASENAME_TO_GLOBAL_INDEX[basename]
