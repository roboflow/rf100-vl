from roboflow import Project
import roboflow
from roboflow.core.version import Version
import os
from typing import Optional, List
from rf100vl.util import get_basename, get_category
import json
from copy import deepcopy


class NotDownloadedError(Exception):
    pass


class RF100VlDataset:
    def __init__(self, rf_project: Project):
        self.rf_project: Project = rf_project
        self.category: str = get_category(get_basename(rf_project.name))
        self.name: str = get_basename(rf_project.name)
        self._location: str = None

    def __repr__(self):
        return f"RF100VLDataset(name={self.name}, category={self.category})"

    def download(self, path: str, model_format: str = "coco", overwrite: bool = True):
        os.makedirs(path, exist_ok=True)
        versions: List[Version] = self.rf_project.versions()
        latest_version: Version = max(versions, key=lambda v: v.id)
        dataset =latest_version.download(
            location=path, model_format=model_format, overwrite=overwrite
        )
        if model_format == "coco":
            self.clean_coco_dataset(path)
        
        self._location = dataset.location

    @property
    def is_fewshot(self) -> str:
        return self.name.contains("fsod")

    @property
    def location(self) -> Optional[str]:
        if self._location is None:
            raise NotDownloadedError("Dataset has not been downloaded yet")
        return self._location

    def clean_coco_dataset(self, path: str) -> None:
        """
        Get clean annotation data: Removes class 0 that is added by default to Roboflow datasets and shifts annotation ids by 1
        """
        for split in ["train", "test", "valid"]:
            anno_path = os.path.join(path, split, "_annotations.coco.json")
            if not os.path.exists(anno_path):
                continue
            with open(anno_path, "r") as f:
                data_ann = json.load(f)
            clean_ann_data = self.get_clean_ann_data(data_ann)
            with open(anno_path, "w") as f:
                json.dump(clean_ann_data, f)

    def get_clean_ann_data(self, data_ann):
        new_data_ann = {}
        if data_ann["info"]:
            new_data_ann["info"] = data_ann["info"]
        if data_ann["licenses"]:
            new_data_ann["licenses"] = data_ann["licenses"]

        # confirm if category 0 is none
        if data_ann["categories"][0]["supercategory"] != "none":
            return

        new_data_ann["categories"] = [
            {
                "id": cat["id"] - 1,
                "name": cat["name"],
                "supercategory": cat["supercategory"],
            }
            for cat in data_ann["categories"]
            if cat["id"] != 0
        ]

        new_data_ann["images"] = data_ann["images"]
        new_data_ann["annotations"] = deepcopy(data_ann["annotations"])

        for ann in new_data_ann["annotations"]:
            ann["category_id"] = ann["category_id"] - 1

        return new_data_ann
