# Image Pose Edit

## Project Overview

This project aims to develop a user-friendly pose edit functionality for images. Just an experiment trying to learn something. The project is divided into two main tasks:

1. Object Segmentation: Segment an object in a given scene based on a user-provided class prompt.
2. Pose Editing: Edit the pose of the segmented object by applying user-specified rotations.

## Current Progress

### Task 1: Object Segmentation

- **Status**: Implemented and functional
- **Functionality**: Segments the specified object in an image and marks it with a red mask
- **Limitations**: Detection accuracy depends on the YOLO model's training dataset. Some objects may not be detected in certain sample images.

### Task 2: Pose Editing

- **Status**: In progress
- **Estimated Completion**: Ongoing, timeline uncertain

## Setup Instructions
### Installation

1. Clone the repository
2. Create and activate a virtual environment
3. Install pre-commit:
   ```
   pip install pre-commit
   pre-commit install
   ```
4. Install pre-commit hooks:
   ```
   pre-commit install --hook-type commit-msg
   pre-commit install --hook-type prepare-commit-msg
   pre-commit autoupdate
   pre-commit run --all-files
   ```
6. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Requirements

- The `requirements.txt` file contains all necessary Python packages.
- Install dependencies using:
  ```
  pip install -r requirements.txt
  ```

### Model Checkpoints

- Model checkpoints are automatically downloaded and handled by the script.
- SAM (Segment Anything Model) checkpoint: ~2.5 GB
- Zero123 checkpoint (for Task 2, under consideration): ~14 GB
- Ensure sufficient disk space before running the script.

## Usage

### Object Segmentation

Run the following command:

```
python run.py --image ./path/to/image.jpg --class "object_name" --output ./path/to/output.png
```

Replace `./path/to/image.jpg` with the input image path, `"object_name"` with the class to segment, and `./path/to/output.png` with the desired output path.

## How It Works

### Task 1: Object Segmentation

1. **SAM (Segment Anything Model)**: Used for initial segmentation of the image.
2. **YOLO (You Only Look Once)**: Employed for object detection and classification.
3. Process:
   - YOLO detects and classifies objects in the image.
   - SAM segments the detected boxes.
   - The script combines SAM's segmentation with YOLO's detection to precisely identify and mask the requested object.

### Task 2: Pose Editing (In Progress)

- Currently exploring the use of Zero123 for 3D-aware image editing.
- It's going to take me a while to understand Zero123's codebase.
- This task is still under development.

## Results
### Input Image
![Input Image](images/chair.jpg)

### Output Image
[![chair-masked.jpg](https://i.postimg.cc/htz575C5/chair-masked.jpg)](https://postimg.cc/xXQRwtVv)

### Command Used
```bash
python run.py --image images/chair.jpg --class "chair" --output output/chair-masked.jpg
```

## Challenges

- Limited object detection capabilities due to the YOLO model's training dataset
- Ongoing development of the pose editing functionality

## Future Work

- Complete implementation of Task 2 (Pose Editing)
- Improve object detection capabilities

## Notes

- The script was developed and tested on a Windows environment.
- pre-commit is used for better code organization and consistency (can skip if you want to).
- The system automatically handles model downloads and initialization.
- First-time run may take longer due to model downloads.
- Ensure a stable internet connection for initial setup.

## References and Acknowledgements
1. Segment Anything Model (SAM) by Meta AI Research
2. YOLO (You Only Look Once) by Ultralytics
3. Zero-1-to-3 by Columbia Vision and Graphics Center
4. pre-commit for maintaining code quality and consistency
5. Python and its extensive ecosystem of libraries and tools

Super grateful to the open-source community and the researchers behind these incredible technologies that enable advancements in computer vision and image processing. There's a lot to learn for beginners like me.
