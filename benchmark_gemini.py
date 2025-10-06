import os
import json
import time
from PIL import Image
from tqdm import tqdm
import argparse
import google.generativeai as genai
from dotenv import load_dotenv


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


def run_gemini_inference(model, image_path, category_names, max_retries=3):
    """Run Gemini 2.5 Pro inference on a single image with zero-shot object detection."""

    # Build the prompt
    prompt = f"""You are given an image. Perform zero-shot object detection.

Task:
- Detect objects in the image that match any of the provided class names.
- Return bounding boxes in [x_min, y_min, x_max, y_max] pixel coordinates.
- Return results as JSON with fields: {{"class": str, "box": [int, int, int, int], "confidence": float}}

Candidate class names:
{json.dumps(category_names)}

Now, analyze the input image and output only the JSON list of detections.
"""

    try:
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Retry logic for rate limits
        for attempt in range(max_retries):
            try:
                # Generate content
                response = model.generate_content([prompt, image])

                # Parse response
                response_text = response.text.strip()

                # Try to extract JSON from response
                # Sometimes the model wraps JSON in markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()

                # Parse JSON
                detections = json.loads(response_text)

                # Ensure detections is a list
                if isinstance(detections, dict):
                    detections = [detections]

                return detections

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Error processing {image_path} after {max_retries} attempts: {e}")
                    return []

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return []


def convert_to_coco_format(detections, category_names, image_id):
    """Convert Gemini detections to COCO format."""
    predictions = []

    # Create category name to ID mapping
    category_name_to_id = {name: idx for idx, name in enumerate(category_names)}

    for detection in detections:
        try:
            class_name = detection.get('class', '')
            box = detection.get('box', [])
            confidence = detection.get('confidence', 0.0)

            # Find category ID
            if class_name not in category_name_to_id:
                continue

            category_id = category_name_to_id[class_name]

            # Convert from [x_min, y_min, x_max, y_max] to COCO format [x, y, width, height]
            if len(box) == 4:
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min

                predictions.append({
                    "image_id": image_id,
                    "bbox": [x_min, y_min, width, height],
                    "score": confidence,
                    "category_id": category_id
                })

        except Exception as e:
            print(f"Error converting detection: {e}")
            continue

    return predictions


def benchmark_dataset(dataset_path, output_dir, model, rate_limit_delay=1.0):
    """Benchmark Gemini 2.5 Pro on a single dataset."""
    dataset_name = os.path.basename(dataset_path)
    test_dir = os.path.join(dataset_path, "test")
    annotation_path = os.path.join(test_dir, "_annotations.coco.json")

    if not os.path.exists(annotation_path):
        print(f"Skipping {dataset_name}: No annotations found at {annotation_path}")
        return

    print(f"\nProcessing dataset: {dataset_name}")

    # Load annotations
    coco_data, categories, category_names, images = load_coco_annotations(annotation_path)
    print(f"Categories: {category_names}")

    # Run inference on all images
    all_predictions = []

    for img_id, img_info in tqdm(images.items(), desc=f"Processing {dataset_name}"):
        image_path = os.path.join(test_dir, img_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Run inference
        detections = run_gemini_inference(model, image_path, category_names)

        # Convert to COCO format
        predictions = convert_to_coco_format(detections, category_names, img_id)
        all_predictions.extend(predictions)

        # Rate limiting
        time.sleep(rate_limit_delay)

    # Save predictions in COCO format
    output_dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dataset_dir, exist_ok=True)
    output_path = os.path.join(output_dataset_dir, "predictions.json")

    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    print(f"Saved {len(all_predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gemini 2.5 Pro on RF100-VL datasets")
    parser.add_argument("--rf100_dir", type=str, default="rf100-vl/rf100-vl",
                        help="Path to RF100-VL datasets directory")
    parser.add_argument("--output_dir", type=str, default="predictions/gemini",
                        help="Directory to save predictions")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-pro",
                        help="Gemini model name")
    parser.add_argument("--rate_limit_delay", type=float, default=1.0,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                        help="Specific datasets to benchmark (default: all)")
    parser.add_argument("--env_file", type=str, default=".env",
                        help="Path to .env file with GOOGLE_AI_API_KEY")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file)
    api_key = os.getenv("GOOGLE_AI_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_AI_API_KEY not found in environment variables or .env file")

    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model_name)
    print(f"Loaded model: {args.model_name}")

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
                rate_limit_delay=args.rate_limit_delay
            )
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
