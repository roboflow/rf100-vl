import os
import sys
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
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


def run_llmdet_inference(model, processor, image_path, text_queries, threshold=0.1):
    """Run LLMDet inference on a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return []

    # Prepare text in format: "class1 . class2 . class3 ."
    text = " . ".join(text_queries) + " ."

    # Prepare inputs
    inputs = processor(images=image, text=text, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]]
    )

    # Extract predictions
    predictions = []
    if len(results) > 0:
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]

        for box, score, label in zip(boxes, scores, labels):
            box = box.cpu().tolist()
            # Convert from [xmin, ymin, xmax, ymax] to COCO format [x, y, width, height]
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            # Map label string back to category_id
            # The label is the text from text_queries
            try:
                category_id = text_queries.index(label)
            except ValueError:
                # If exact match fails, try to find closest match
                category_id = 0
                for i, query in enumerate(text_queries):
                    if query.lower() in label.lower() or label.lower() in query.lower():
                        category_id = i
                        break

            predictions.append({
                "bbox": [x_min, y_min, width, height],
                "score": score.item() if torch.is_tensor(score) else float(score),
                "category_id": category_id
            })

    return predictions


def benchmark_dataset(dataset_path, output_dir, model_name="iSEE-Laboratory/llmdet_large", threshold=0.1, device="cuda", gpu_id=None):
    """Benchmark LLMDet on a single dataset."""
    dataset_name = os.path.basename(dataset_path)
    test_dir = os.path.join(dataset_path, "test")
    annotation_path = os.path.join(test_dir, "_annotations.coco.json")

    if not os.path.exists(annotation_path):
        print(f"Skipping {dataset_name}: No annotations found at {annotation_path}")
        return

    print(f"\nProcessing dataset: {dataset_name}")

    # Load annotations
    coco_data, categories, category_names, images = load_coco_annotations(annotation_path)

    # Prepare text queries (simple category names for zero-shot)
    text_queries = category_names
    print(f"Categories: {category_names}")

    # Load model
    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)

    # Move model to device
    if device == "cuda" and torch.cuda.is_available():
        if gpu_id is not None:
            device = f"cuda:{gpu_id}"
            model = model.to(device)
        else:
            model = model.cuda()
    else:
        model = model.cpu()
        device = "cpu"

    model.eval()

    # Run inference on all images
    all_predictions = []

    for img_id, img_info in tqdm(images.items(), desc=f"Processing {dataset_name}"):
        image_path = os.path.join(test_dir, img_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        predictions = run_llmdet_inference(model, processor, image_path, text_queries, threshold)

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
    parser = argparse.ArgumentParser(description="Benchmark LLMDet on RF100-VL datasets")
    parser.add_argument("--rf100_dir", type=str, default="rf100-vl/rf100-vl",
                        help="Path to RF100-VL datasets directory")
    parser.add_argument("--output_dir", type=str, default="predictions/llmdet",
                        help="Directory to save predictions")
    parser.add_argument("--model_name", type=str, default="iSEE-Laboratory/llmdet_large",
                        help="LLMDet model name (llmdet_large, llmdet_base, llmdet_tiny)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="Specific GPU ID to use (for multi-GPU runs)")
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
                model_name=args.model_name,
                threshold=args.threshold,
                device=args.device,
                gpu_id=args.gpu_id
            )
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
