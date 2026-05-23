import os
import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_predictions(dataset_path, predictions_path):
    """Evaluate predictions against ground truth using COCO metrics."""
    dataset_name = os.path.basename(dataset_path)
    annotation_path = os.path.join(dataset_path, "test", "_annotations.coco.json")

    if not os.path.exists(annotation_path):
        print(f"Error: Ground truth not found at {annotation_path}")
        return None

    if not os.path.exists(predictions_path):
        print(f"Error: Predictions not found at {predictions_path}")
        return None

    print(f"\n{'='*80}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*80}")

    # Load ground truth
    coco_gt = COCO(annotation_path)

    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    if len(predictions) == 0:
        print("Warning: No predictions to evaluate!")
        return None

    # Load predictions into COCO format
    coco_dt = coco_gt.loadRes(predictions)

    # Evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Return metrics
    metrics = {
        "dataset": dataset_name,
        "AP@[.5:.95]": coco_eval.stats[0],
        "AP@.50": coco_eval.stats[1],
        "AP@.75": coco_eval.stats[2],
        "AP@[.5:.95]_small": coco_eval.stats[3],
        "AP@[.5:.95]_medium": coco_eval.stats[4],
        "AP@[.5:.95]_large": coco_eval.stats[5],
        "AR@[.5:.95]_max1": coco_eval.stats[6],
        "AR@[.5:.95]_max10": coco_eval.stats[7],
        "AR@[.5:.95]_max100": coco_eval.stats[8],
        "AR@[.5:.95]_small": coco_eval.stats[9],
        "AR@[.5:.95]_medium": coco_eval.stats[10],
        "AR@[.5:.95]_large": coco_eval.stats[11],
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate OWLv2 predictions")
    parser.add_argument("--rf100_dir", type=str, default="rf100-vl/rf100-vl",
                        help="Path to RF100-VL datasets directory")
    parser.add_argument("--predictions_dir", type=str, default="predictions/owlv2",
                        help="Directory containing predictions")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                        help="Specific datasets to evaluate (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Get list of datasets to evaluate
    if args.datasets:
        datasets = args.datasets
    else:
        # Find all datasets with predictions
        if os.path.exists(args.predictions_dir):
            datasets = [d for d in os.listdir(args.predictions_dir)
                       if os.path.isdir(os.path.join(args.predictions_dir, d))]
        else:
            print(f"Error: Predictions directory not found: {args.predictions_dir}")
            return

    print(f"Found {len(datasets)} datasets to evaluate")

    # Evaluate each dataset
    all_results = []

    for dataset_name in datasets:
        dataset_path = os.path.join(args.rf100_dir, dataset_name)
        predictions_path = os.path.join(args.predictions_dir, dataset_name, "predictions.json")

        try:
            metrics = evaluate_predictions(dataset_path, predictions_path)
            if metrics:
                all_results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<60} {'AP@[.5:.95]':>10} {'AP@.50':>10}")
    print("-" * 80)

    for result in all_results:
        print(f"{result['dataset']:<60} {result['AP@[.5:.95]']:>10.3f} {result['AP@.50']:>10.3f}")

    if all_results:
        avg_ap = sum(r["AP@[.5:.95]"] for r in all_results) / len(all_results)
        avg_ap50 = sum(r["AP@.50"] for r in all_results) / len(all_results)
        print("-" * 80)
        print(f"{'AVERAGE':<60} {avg_ap:>10.3f} {avg_ap50:>10.3f}")

    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
