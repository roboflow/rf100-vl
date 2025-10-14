import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import sys

# Disable torch.compile to avoid device mismatch errors
torch._dynamo.config.suppress_errors = True
os.environ['TORCH_COMPILE_DISABLE'] = '1'

sys.path.insert(0, '/root/sam3')

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


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


def run_sam3_inference(model, processor, image_path, text_queries, threshold=0.5):
    """Run SAM3 inference on a single image with text prompts."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return []

    width, height = image.size
    all_predictions = []

    # Run inference for each text query (category)
    for category_idx, text_query in enumerate(text_queries):
        try:
            # Initialize inference state
            inference_state = processor(image, instance_prompt=False)

            # Add text prompt
            processor.add_prompt(inference_state, text_str=text_query, instance_prompt=False)

            # Run inference
            model.run_inference(inference_state)

            # Get outputs
            out = processor.postprocess_output(inference_state, output_prob_thresh=threshold)

            # Extract predictions
            if 'out_binary_masks' in out and 'out_boxes_xywh' in out and 'out_probs' in out:
                masks = out['out_binary_masks']
                boxes_xywh = out['out_boxes_xywh']
                probs = out['out_probs']

                for box, prob in zip(boxes_xywh, probs):
                    # box is in normalized xywh format, convert to absolute
                    x, y, w, h = box
                    x_abs = x * width
                    y_abs = y * height
                    w_abs = w * width
                    h_abs = h * height

                    all_predictions.append({
                        "bbox": [float(x_abs), float(y_abs), float(w_abs), float(h_abs)],
                        "score": float(prob),
                        "category_id": category_idx + 1  # 1-indexed to match COCO format
                    })

            # Reset state for next query
            processor.reset_state(inference_state)

        except Exception as e:
            print(f"Error processing text query '{text_query}': {e}")
            continue

    return all_predictions


def benchmark_dataset(dataset_path, output_dir, bpe_path, checkpoint_path,
                      has_presence_token=True, threshold=0.5, device="cuda", gpu_id=None):
    """Benchmark SAM3 on a single dataset."""
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

    # Load model
    print(f"Loading SAM3 model from {checkpoint_path}")

    # Set device
    if device == "cuda" and torch.cuda.is_available():
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        has_presence_token=has_presence_token
    )

    # Enable autocast for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processor = Sam3Processor()

    # Run inference on all images
    all_predictions = []

    for img_id, img_info in tqdm(images.items(), desc=f"Processing {dataset_name}"):
        image_path = os.path.join(test_dir, img_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = run_sam3_inference(model, processor, image_path, text_queries, threshold)

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
    parser = argparse.ArgumentParser(description="Benchmark SAM3 on RF100-VL datasets")
    parser.add_argument("--rf100_dir", type=str, default="/root/rf100-vl-data",
                        help="Path to RF100-VL datasets directory")
    parser.add_argument("--output_dir", type=str, default="/root/predictions/sam3",
                        help="Directory to save predictions")
    parser.add_argument("--bpe_path", type=str, default="/root/bpe_simple_vocab_16e6.txt.gz",
                        help="Path to BPE vocabulary file")
    parser.add_argument("--checkpoint_path", type=str, default="/root/weights.pt",
                        help="Path to SAM3 checkpoint")
    parser.add_argument("--has_presence_token", action="store_true", default=True,
                        help="Whether the model has presence token")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to run inference on")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU ID to use (if multiple GPUs available)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to benchmark (default: all)")

    args = parser.parse_args()

    # Get list of datasets
    rf100_dir = args.rf100_dir
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [d for d in os.listdir(rf100_dir)
                   if os.path.isdir(os.path.join(rf100_dir, d))]
        datasets.sort()

    print(f"Found {len(datasets)} datasets to process")

    # Process each dataset
    for dataset_name in datasets:
        dataset_path = os.path.join(rf100_dir, dataset_name)
        try:
            benchmark_dataset(
                dataset_path=dataset_path,
                output_dir=args.output_dir,
                bpe_path=args.bpe_path,
                checkpoint_path=args.checkpoint_path,
                has_presence_token=args.has_presence_token,
                threshold=args.threshold,
                device=args.device,
                gpu_id=args.gpu_id
            )
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
