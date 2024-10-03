import argparse
import os

from settings import MODEL_CONFIG
from utils.sam_utils import (
    download_if_model_not_exists,
    download_zero123_checkpoints,
    generate_masks,
    load_sam,
    save_masked_image,
)


# Define the function to handle command line arguments
# open to future modifications
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM on a given image.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--class",
        type=str,
        required=True,
        help="Object class to segment (e.g., chair).",
        dest="object_class",
    )

    # parser.add_argument(
    #     "--azimuth",
    #     type=float,
    #     help="Azimuth angle change (optional).",
    # )

    # parser.add_argument(
    #     "--polar",
    #     type=float,
    #     help="Polar angle change (optional).",
    # )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output image.",
    )
    return parser.parse_args()


# Get the model config values
model_name = MODEL_CONFIG["name"]
model_path = MODEL_CONFIG["path"]
model_url = MODEL_CONFIG["url"]

# Download the model if it doesn't already exist
download_if_model_not_exists(model_path, model_url)

# Download Zero123
# iteration = MODEL_CONFIG["zero123_iteration"]
# download_zero123_checkpoints(iteration)


def main():
    args = parse_args()

    # Ensure the path passed (via args) are valid
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file {args.image} not found.")

    # Load SAM
    print("Loading SAM...")
    sam = load_sam(model_name, model_path)

    masks = generate_masks(sam, args.image, 1, args.object_class)

    print(masks)

    save_masked_image(args.image, masks, args.output)


if __name__ == "__main__":
    main()
