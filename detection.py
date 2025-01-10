import torch
import cv2
import json
import os

# Initialize YOLOv5 model
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create a directory for output if it doesn't exist
os.makedirs("output", exist_ok=True)

def analyze_video(input_video, output_data):
    # Verify if the input video exists
    if not os.path.isfile(input_video):
        print(f"Error: The specified file '{input_video}' was not found.")
        return

    # Load the video
    video_stream = cv2.VideoCapture(input_video)
    if not video_stream.isOpened():
        print(f"Error: Unable to access the video file '{input_video}'.")
        return

    detections = []
    current_frame = 0

    print("Analyzing video frames...")

    while video_stream.isOpened():
        success, frame = video_stream.read()
        if not success:
            break

        print(f"Analyzing frame {current_frame}...")

        # Detect objects in the frame
        detection_results = detector(frame)
        frame_detections = detection_results.pandas().xyxy[0]

        # Structure detection information for each object
        for _, detection in frame_detections.iterrows():
            detection_entry = {
                "type": detection['name'],
                "frame": current_frame,
                "bounding_box": [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']],
                "nested_info": {}  # Placeholder for additional details
            }
            detections.append(detection_entry)

        current_frame += 1

    # Release video resources
    video_stream.release()

    # Export the detections to a JSON file
    with open(output_data, "w") as result_file:
        json.dump(detections, result_file, indent=4)

    print(f"Detection data has been saved to '{output_data}'")

if __name__ == "__main__":
    # Define paths for video input and JSON output
    input_file = "test_video.mp4"  # Ensure this file is present in the working directory
    json_output = "output/detections.json"

    # Run the video analysis and save results
    analyze_video(input_file, json_output)
