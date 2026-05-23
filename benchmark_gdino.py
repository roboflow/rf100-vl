import os
import sys
import json
import torch
from PIL import Image
from tqdm import tqdm
import argparse

# Add GroundingDINO to path
sys.path.append('/root/GroundingDINO')

from groundingdino.util.inference import load_model, load_image, predict


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


def run_gdino_inference(model, image_path, text_queries, threshold=0.1):
    """Run Grounding DINO inference on a single image."""
    try:
        # Load image
        image_source, image = load_image(image_path)

        # Prepare text prompt (Grounding DINO format: "class1 . class2 . class3")
        text_prompt = " . ".join(text_queries) + " ."

        # Run inference
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=threshold,
            text_threshold=threshold
        )

        # Extract predictions
        predictions = []
        h, w = image_source.shape[:2]

        for box, score, phrase in zip(boxes, logits, phrases):
            # Convert from normalized [cx, cy, w, h] to [x, y, width, height]
            cx, cy, box_w, box_h = box.tolist()
            x = (cx - box_w / 2) * w
            y = (cy - box_h / 2) * h
            width = box_w * w
            height = box_h * h

            # Map phrase to category_id
            phrase_lower = phrase.lower().strip()
            category_id = 0
            for i, query in enumerate(text_queries):
                if query.lower() in phrase_lower or phrase_lower in query.lower():
                    category_id = i
                    break

            predictions.append({
                "bbox": [x, y, width, height],
                "score": score.item(),
                "category_id": category_id
            })

        return predictions

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []


def benchmark_dataset(dataset_path, output_dir, model, threshold=0.1):
    """Benchmark Grounding DINO on a single dataset."""
    dataset_name = os.path.basename(dataset_path)
    test_dir = os.path.join(dataset_path, "test")
    annotation_path = os.path.join(test_dir, "_annotations.coco.json")

    if not os.path.exists(annotation_path):
        print(f"Skipping {dataset_name}: No annotations found at {annotation_path}")
        return

    print(f"\nProcessing dataset: {dataset_name}")

    # Load annotations
    coco_data, categories, category_names, images = load_coco_annotations(annotation_path)

    # Prepare text queries
    text_queries = category_names
    print(f"Categories: {category_names}")

    # Run inference on all images
    all_predictions = []

    for img_id, img_info in tqdm(images.items(), desc=f"Processing {dataset_name}"):
        image_path = os.path.join(test_dir, img_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        predictions = run_gdino_inference(model, image_path, text_queries, threshold)

        # Add image_id to each prediction
        for pred in predictions:
            pred['image_id'] = img_id

        all_predictions.extend(predictions)

    # Save predictions in COCO format
    output_dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dataset_dir, exist_ok=True)
    output_path = os.path.join(output_dataset_dir, "predictions.json")

    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    print(f"Saved {len(all_predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Grounding DINO on RF100-VL datasets")
    parser.add_argument("--rf100_dir", type=str, default="rf100-vl/rf100-vl",
                        help="Path to RF100-VL datasets directory")
    parser.add_argument("--output_dir", type=str, default="predictions/gdino",
                        help="Directory to save predictions")
    parser.add_argument("--config", type=str,
                        default="/root/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="Path to Grounding DINO config file")
    parser.add_argument("--weights", type=str,
                        default="/root/GroundingDINO/weights/groundingdino_swint_ogc.pth",
                        help="Path to Grounding DINO weights")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                        help="Specific datasets to benchmark (default: all)")

    args = parser.parse_args()

    # Check if weights exist
    if not os.path.exists(args.weights):
        print(f"Error: Weights not found at {args.weights}")
        print("Please download weights from:")
        print("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        return

    # Load model
    print(f"Loading Grounding DINO model...")
    model = load_model(args.config, args.weights, device=args.device)

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
                model=model,
                threshold=args.threshold
            )
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
