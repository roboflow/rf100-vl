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
