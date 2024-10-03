import os
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)
from tqdm import tqdm
from ultralytics import YOLO

from settings import MODEL_CONFIG


def download_if_model_not_exists(model_path, model_url):
    """
    Checks if the model exists locally, and if not, download it.
    """
    if not os.path.exists(model_path):
        print(
            f"Model not found at {model_path}; downloading from {model_url}..."
        )

        # Create a directory if it doesn't already exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        tqdm_params = {
            "unit": "B",
            "unit_scale": True,
            "desc": model_path,
            "miniters": 1,
            "unit_divisor": 1024,
        }

        def tqdm_hook(t):
            last_b = [0]

            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b

            return update_to

        # Enable a progress bar for downloading, since the file size is too large
        with tqdm(**tqdm_params) as t:
            urllib.request.urlretrieve(
                model_url, model_path, reporthook=tqdm_hook(t)
            )

            print(
                f"\nModel successfully downloaded and saved at {model_path}!"
            )
    else:
        print(f"Model already exists at {model_path}. Using it.")


def load_sam(model_name, model_path):
    """
    Load the SAM model from a model type and model path.
    """
    sam = sam_model_registry[model_name](checkpoint=model_path)

    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name()} for computation.")
        sam = sam.cuda()
    else:
        print("Using CPU since no CUDA hardware was found.")
        sam = sam.to("cpu")

    return sam


def detect_object(image, object_class):
    """
    Detect object of the specified class using YOLOv8 and return a list of bounding boxes.
    """
    yolo_model_path = MODEL_CONFIG["yolo_model"]
    yolo_model = YOLO(yolo_model_path)
    results = yolo_model(image, conf=0.5, device="cuda:0")  # Perform inference

    detected_objects = []  # To store detected objects of the specified class

    # Iterate over results and detect the class
    for result in results:
        boxes = result.boxes  # Boxes object (no need for .cpu() or .numpy())

        for box in boxes:
            class_id = int(box.cls[0])  # Get class ID of the detected object
            if yolo_model.names[class_id] == object_class:  # Match class name
                detected_objects.append(
                    box.xyxy.tolist()
                )  # Add bounding box (convert to list)

    # Return detected objects or None if nothing is found
    if not detected_objects:
        print(
            f"\n\nNo {object_class} detected. Here are the objects that were detected:"
        )
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                print(f"\t- {yolo_model.names[class_id]}")

    return detected_objects if detected_objects else None


def generate_masks(sam, image_path, mode=0, object_class=None):
    """
    Generate masks for an image using the SAM model.

    Args:
        sam: The loaded SAM model.
        image_path: Path to the input image.
        mode: 0 for automatic mask generation, else prompt-based generation
        object_class: Optional; class to filter masks (used in prompt-based mode).

    Returns:
        masks: List of generated masks.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mode == 0:
        generator = SamAutomaticMaskGenerator(sam)
        masks = generator.generate(image)
    else:
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        if object_class:
            bbox = detect_object(image, object_class)
            if bbox is not None:
                bbox = np.array([bbox])
                masks, _, _ = predictor.predict(box=bbox)
            else:
                print(f"\nNo {object_class} detected in the image. Exiting.")
                return None
        else:
            print(
                "Please provide an object class for prompt-based segmentation."
            )
            return None

    return masks


def save_masked_image(original_image_path, masks, output_image_path):
    # If no masks are found, save the original image
    if masks is None:
        print("\nNo masks to apply. Saving original image.")
        original_image = cv2.imread(original_image_path)
        cv2.imwrite(output_image_path, original_image)
        return

    # Read the original image
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Create a transparent RGBA image for the masks
    height, width = original_image.shape[:2]
    mask_image = np.zeros((height, width, 4), dtype=np.float32)

    # Sort masks by area
    if isinstance(masks, list) and "area" in masks[0]:
        sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    else:
        sorted_masks = [{"segmentation": masks[0], "area": np.sum(masks[0])}]

    # Apply each mask with red color
    red_color = np.array([1.0, 0.0, 0.0, 0.5])  # Red color with 50% opacity
    for mask in sorted_masks:
        mask_image[mask["segmentation"]] = red_color

    # Apply each mask with a random color
    # for mask in sorted_masks:
    #     color = np.concatenate([np.random.random(3), [0.35]])  # Random color with 35% opacity
    #     mask_image[mask['segmentation']] = color

    # Convert mask_image to uint8 and remove alpha channel
    mask_image_rgb = (mask_image[:, :, :3] * 255).astype(np.uint8)

    # Blend the original image with the mask image
    blended_image = cv2.addWeighted(
        original_image, 0.7, mask_image_rgb, 0.3, 0
    )

    # Create a figure and display the blended image
    plt.figure(figsize=(10, 10))
    plt.imshow(blended_image)
    plt.axis("off")

    # Save the figure
    plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Masked image saved at {output_image_path}")
