
---

# Object Detection with OpenCV and TensorFlow

## Overview

This project implements an object detection system using OpenCV for image processing and TensorFlow for model prediction. The system reads an image, processes it to detect specific objects based on color thresholds, and then uses a pre-trained model to classify the detected objects.

## Features

- Color-based object detection using HSV color space.
- Contour detection and bounding box drawing for identified objects.
- Integration with a TensorFlow Keras model for object classification.
- Resizing of detected object images for model input.
- Simple visualization of original and processed images.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Additional libraries as needed (e.g., Matplotlib for plotting, if used)

You can install the required libraries using pip:

```bash
pip install opencv-python numpy tensorflow
```

## Usage

1. **Model Path**: Ensure you have a trained TensorFlow Keras model saved at `path/to/model`.
2. **Image Path**: Update the paths in the `main` section of the script to point to the images you want to process.

```python
paths = ("path/to/image")
```

3. **Run the Script**: Execute the script to perform detections and display the results.

```bash
python detection.py
```

## Class Structure

### `Detection`

- **`__init__(self, path)`**: Initializes the Detection class. Loads the model and the image at the specified path, and converts the image to HSV color space.
  
- **`show_images(self)`**: Displays the original and processed images.

- **`reducing_picture(self)`**: Resizes the original and edge-detected images for display.

- **`detections(self)`**: Processes the image to detect contours, applies color masks, and uses the model to classify the detected objects. Draws bounding boxes around identified objects.

- **`__del__(self)`**: Cleans up by closing any OpenCV windows.

## Error Handling

The script includes basic error handling. Any exceptions raised during execution will be caught and printed to the console.

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
