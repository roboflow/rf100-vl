import os
import json
import time
from PIL import Image
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from tqdm import tqdm
import argparse


def load_coco_annotations(annotation_path):
    """Load COCO format annotations and extract categories."""
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Extract category names
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_names = [cat['name'] for cat in coco_data['categories']]

    # Extract images
    images = {img['id']: img for img in coco_data['images']}

    return coco_data, categories, category_names, images


def run_dinox_inference(client, image_path, text_prompt, threshold=0.25, iou_threshold=0.8):
    """Run DINO-X inference on a single image via API."""
    try:
        # Convert image to base64
        image = image_to_base64(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return []

    # Prepare API request
    api_path = "/v2/task/dinox/detection"
    api_body = {
        "model": "DINO-X-1.0",
        "image": image,
        "prompt": {
            "type": "text",
            "text": text_prompt
        },
        "targets": ["bbox"],
        "bbox_threshold": threshold,
        "iou_threshold": iou_threshold
    }

    task = V2Task(api_path=api_path, api_body=api_body)

    try:
        client.run_task(task)
        result = task.result
        objects = result.get("objects", [])
    except Exception as e:
        print(f"Error running inference on {image_path}: {e}")
        return []

    # Convert results to COCO format
    predictions = []
    for obj in objects:
        bbox = obj["bbox"]  # Already in COCO format [x, y, width, height]
        score = obj["score"]
        category_name = obj["category"].lower().strip()

        predictions.append({
            "bbox": bbox,
            "score": score,
            "category_name": category_name
        })

    return predictions


def benchmark_dataset(dataset_path, output_dir, api_token, threshold=0.25, iou_threshold=0.8, rate_limit_delay=0.5):
    """Benchmark DINO-X on a single dataset."""
    dataset_name = os.path.basename(dataset_path)
    test_dir = os.path.join(dataset_path, "test")
    annotation_path = os.path.join(test_dir, "_annotations.coco.json")

    if not os.path.exists(annotation_path):
        print(f"Skipping {dataset_name}: No annotations found at {annotation_path}")
        return

    print(f"\nProcessing dataset: {dataset_name}")

    # Load annotations
    coco_data, categories, category_names, images = load_coco_annotations(annotation_path)

    # Create name to id mapping
    category_name_to_id = {name.lower().strip(): cat_id for cat_id, name in categories.items()}

    # Prepare text prompt in DINO-X format (period-separated)
    text_prompt = " . ".join(category_names)
    print(f"Categories: {category_names}")
    print(f"Text prompt: {text_prompt}")

    # Initialize DINO-X client
    config = Config(api_token)
    client = Client(config)

    # Run inference on all images
    all_predictions = []

    for img_id, img_info in tqdm(images.items(), desc=f"Processing {dataset_name}"):
        image_path = os.path.join(test_dir, img_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        predictions = run_dinox_inference(
            client, image_path, text_prompt, threshold, iou_threshold
        )

        # Map category names to IDs and add image_id
        for pred in predictions:
            category_name = pred["category_name"]
            if category_name in category_name_to_id:
                pred['category_id'] = category_name_to_id[category_name]
                pred['image_id'] = img_id
                # Remove the category_name field as it's not needed in COCO format
                del pred['category_name']
                all_predictions.append(pred)
            else:
                print(f"Warning: Category '{category_name}' not found in dataset categories")

        # Rate limiting to avoid API throttling
        time.sleep(rate_limit_delay)

    # Save predictions in COCO format
    output_dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dataset_dir, exist_ok=True)
    output_path = os.path.join(output_dataset_dir, "predictions.json")

    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    print(f"Saved {len(all_predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DINO-X on RF100-VL datasets")
    parser.add_argument("--rf100_dir", type=str, default="rf100-vl/rf100-vl",
                        help="Path to RF100-VL datasets directory")
    parser.add_argument("--output_dir", type=str, default="../predictions/dinox",
                        help="Directory to save predictions")
    parser.add_argument("--api_token", type=str, required=True,
                        help="DINO-X API token")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.8,
                        help="IOU threshold for NMS")
    parser.add_argument("--rate_limit_delay", type=float, default=0.5,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                        help="Specific datasets to benchmark (default: all)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of datasets
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [d for d in os.listdir(args.rf100_dir)
                   if os.path.isdir(os.path.join(args.rf100_dir, d))]

    print(f"Found {len(datasets)} datasets to process")

    # Process each dataset
    for dataset_name in datasets:
        dataset_path = os.path.join(args.rf100_dir, dataset_name)

        if not os.path.isdir(dataset_path):
            print(f"Skipping {dataset_name}: Not a directory")
            continue

        try:
            benchmark_dataset(
                dataset_path=dataset_path,
                output_dir=args.output_dir,
                api_token=args.api_token,
                threshold=args.threshold,
                iou_threshold=args.iou_threshold,
                rate_limit_delay=args.rate_limit_delay
            )
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
