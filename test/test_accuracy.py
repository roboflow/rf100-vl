import os
from rf100vl import download_rf100vl_fsod, get_rf100vl_fsod_projects
from rf100vl.dataset import RF100VlDataset
from rf100vl.roboflow100vl import DatasetList
API_KEY = os.getenv("ROBOFLOW_API_KEY")

def download_one_dataset():
    dataset = get_rf100vl_fsod_projects(API_KEY)[80]
    dataset.download(os.path.join(os.path.dirname(__file__), "datasets", dataset.name))
    print(type(dataset))
    return dataset.location

def test_gt_accuracy_with_self():
    location = download_one_dataset()
    import pycocotools.coco as coco
    from pycocotools.cocoeval import COCOeval

    coco_gt = coco.COCO(os.path.join(location, "train", "_annotations.coco.json"))
    coco_pred = coco.COCO(os.path.join(location, "train", "_annotations.coco.json"))

    # Use all images in the ground truth

    # Use ground truth annotations as predictions (perfect prediction)
    anns = coco_gt.loadAnns(coco_gt.getAnnIds())
    # Convert to COCO detection format
    coco_dt = []
    for ann in anns:
        coco_dt.append({
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "score": 1.0,  # perfect confidence
            "id": ann["id"]
        })

    # Load detections into COCO
    coco_dt = coco_gt.loadRes(coco_dt)

    # Evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Assert mAP is 1.0 (AP at IoU=0.50:0.95)
    assert abs(coco_eval.stats[0] - 1.0) < 1e-6, f"mAP is not 1, got {coco_eval.stats[0]}"


if __name__ == "__main__":
    test_gt_accuracy_with_self()