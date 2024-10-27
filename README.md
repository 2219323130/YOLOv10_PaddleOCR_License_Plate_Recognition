# YOLOv10_PaddleOCR_License_Plate_Recognition 

## Overview

This project aims to develop an automated system for detecting and recognizing license plates from images, videos, and live video feeds. The system leverages state-of-the-art computer vision technologies, specifically YOLOv10n (Nano Architecture used for faster inferencing) for object detection and PaddleOCR for optical character recognition (OCR). 

## Features

- **License Plate Detection**:  Detects license plates using YOLOv10.
- **Optical Character Recognition (OCR)**:  Extracts text from detected license plates using PaddleOCR.
- **Image Processing**:  Detect and recognize the license plate number on the image.
- **Video Processing**:  Processes video files to detect and recognize license plates frame by frame.
- **Live Feed Processing**:  Captures and processes real-time video feeds from a webcam.

## Technologies Used

- **YOLOv10 (You Only Look Once)**:  A deep learning model for real-time object detection.
- **PaddleOCR**:  A practical ultra lightweight OCR system.
- **OS**: A built-in module that provides a way of using operating system-dependent functionality.
- **Streamlit**:  An open-source framework designed for building interactive web applications quickly and easily. 
- **OpenCV**:  A library for image processing and computer vision tasks.
- **NumPy**:  A fundamental library for scientific computing in Python.
- **PIL**: A powerful library for image processing in Python.

## Installation

conda virtual environment is recommended. 
```
conda create -n myenv python=3.10
conda activate myenv
pip install paddlepaddle
pip install paddleocr
pip install streamlit
pip install ultralytics
```

## Usage

### If you just use my trained model

1.Change the model path, the model file is in YOLOv10_PaddleOCR_License_Plate_Recognition\train\weights\best.pt.
2.Change the font_path,the font file is in YOLOv10_PaddleOCR_License_Plate_Recognition\simfang.ttf.

### If you want to train your own license plate detection and recognition model

Dataset can be downloaded at https://universe.roboflow.com/
You can be in https://colab.research.google.com/ for model training  
1.Change the model path.    
2.Change the license_plate_label,you can refer to the image under YOLOv10_PaddleOCR_License_Plate_Recognition\images\license_plate_label.jpg to modify it.   
3.Change the font_path.  

### Running the Application

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```

### Input Sources

1. **Image Processing**: Upload an image file (`.jpg`, `.jpeg`, `.png`) to perform license plate detection and recognition.
2. **Video Processing**: Upload a video file (`.mp4`, `.avi`, `.mov`) to process each frame for license plate detection and recognition.
3. **Live Feed Processing**: Stream video from a webcam to detect and recognize license plates in real time.

### Example

Here is an example of how the application works with an uploaded image:

1. **Upload**: Choose an image file from your local machine.
2. **Processing**: The application detects and recognizes the license plate.
3. **Results**: The processed image with detected license plate(s) and recognized text is displayed.

### Image Processing

![LP-output](https://github.com/2219323130/YOLOv10_PaddleOCR_License_Plate_Recognition/blob/main/images/output_image_processing.jpg)

### Live Feed Processing

![LP-output](https://github.com/2219323130/YOLOv10_PaddleOCR_License_Plate_Recognition/blob/main/images/output_live_feed_processing.jpg)

## Code

The core functionality of the application is implemented in the `app.py` file. Key components include:

- **Object Detection**: Utilizes YOLOv10 for detecting license plates in images and video frames.
- **OCR Processing**: Uses PaddleOCR to extract text from detected license plates.
- **Streamlit Integration**: Provides an interactive web interface for users to upload images/videos and view results.
