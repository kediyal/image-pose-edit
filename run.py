import argparse


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
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
