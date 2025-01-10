
# AI Engineer Assessment Project

This project demonstrates object and sub-object detection in videos with hierarchical JSON output.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision opencv-python numpy
   ```

2. **Run the Script**:
   ```bash
   python detection.py
   ```

3. **Outputs**:
   - JSON output: Saved in the `output/` folder as `detections.json`.
   - Cropped images: Not implemented yet (placeholder in code).

## Notes
- The code uses YOLOv5 for object detection.
- Ensure `test_video.mp4` is present in the project directory.
