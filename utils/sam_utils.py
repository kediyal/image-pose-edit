import os
import urllib.request

from tqdm import tqdm


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


def load_sam():
    pass
