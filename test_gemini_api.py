#!/usr/bin/env python3
"""Quick test script to verify Gemini API connection works."""
import os
import json
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")
api_key = os.getenv("GOOGLE_AI_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_AI_API_KEY not found in environment")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-pro")

# Test with first image from actions dataset
test_image_path = "rf100-vl/actions/test/Aarau-EmmenNord-01_0000_Layer-121_jpg.rf.1262334cbac307efba1465bc083466d1.jpg"

if not os.path.exists(test_image_path):
    print(f"Error: Test image not found at {test_image_path}")
    exit(1)

# Load image
image = Image.open(test_image_path).convert('RGB')
print(f"Loaded test image: {test_image_path}")
print(f"Image size: {image.size}")

# Build prompt
category_names = ["person", "bicycle", "car"]  # Simple test categories
prompt = f"""You are given an image. Perform zero-shot object detection.

Task:
- Detect objects in the image that match any of the provided class names.
- Return bounding boxes in [x_min, y_min, x_max, y_max] pixel coordinates.
- Return results as JSON with fields: {{"class": str, "box": [int, int, int, int], "confidence": float}}

Candidate class names:
{json.dumps(category_names)}

Now, analyze the input image and output only the JSON list of detections.
"""

print("\nSending request to Gemini API...")
response = model.generate_content([prompt, image])

print("\nResponse received:")
print(response.text)

# Try to parse as JSON
response_text = response.text.strip()
if "```json" in response_text:
    json_start = response_text.find("```json") + 7
    json_end = response_text.find("```", json_start)
    response_text = response_text[json_start:json_end].strip()
elif "```" in response_text:
    json_start = response_text.find("```") + 3
    json_end = response_text.find("```", json_start)
    response_text = response_text[json_start:json_end].strip()

print("\nParsed JSON:")
detections = json.loads(response_text)
print(json.dumps(detections, indent=2))

print(f"\nFound {len(detections)} detections")
print("\nAPI test successful!")
