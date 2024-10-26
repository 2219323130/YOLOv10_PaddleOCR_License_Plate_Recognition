import os
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
from paddleocr import PaddleOCR

# the language is set to Chinese, which can recognize Chinese and English at the same time.
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# YOLOv10_PaddleOCR_License_Plate_Recognition\train\weights\best.pt has my trained model, you can replace it with your own trained model.
model = YOLO(r"D:\pythonfiles\YOLOv10_PaddleOCR_License_Plate_Recognition\train\weights\best.pt")

def process_image(image, apply_rotation=True):
    results = model(image)

    # If you train your own model, you will have to change the license_plate_label to the ones used in the training dataset.
    license_plate_label = "license"
    confidence_threshold = 0.5

    if license_plate_label in model.names.values():
        label_index = list(model.names.values()).index(license_plate_label)
    else:
        return image

    if label_index is not None:
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confidences = result.boxes.conf
            for i, cls in enumerate(classes):
                if int(cls) == label_index and confidences[i] > confidence_threshold:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    license_plate_image = image[y1:y2, x1:x2]
                    ocr_result = ocr.ocr(license_plate_image, cls=True)

                    if ocr_result:
                        if isinstance(ocr_result, list) and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
                            license_plate_number = ocr_result[0][0][1][0]
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)

                            # Change the path to the actual path of the font file.
                            font_path = "simfang.ttf"
                            font_size = 50
                            font = ImageFont.truetype(font_path, font_size)
                            pil_image = Image.fromarray(image)
                            draw = ImageDraw.Draw(pil_image)
                            draw.text((x1 - 40, y1 - font_size), f"{license_plate_number.upper()}",
                                       font=font, fill=(255, 0, 0))
                            image = np.array(pil_image)
                        else:
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)

    return image


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            processed_frame = process_image(frame, apply_rotation=False)
            frame_placeholder.image(processed_frame, channels="BGR")
            out.write(processed_frame)
        except :
            continue

    cap.release()
    out.release()

def process_live_feed():
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            processed_frame = process_image(frame, apply_rotation=False)
            frame_placeholder.image(processed_frame, channels="BGR")
        except:
            continue

    cap.release()

st.title("License Plate detection and Recognition system")
option = st.sidebar.selectbox("Choose your type", ("Image Processing", "Video Processing", "Live Feed Processing"))

if option == "Image Processing":
    uploaded_file = st.file_uploader("Enter an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        processed_image = process_image(image)
        st.image(processed_image, caption='Processed Image with License Plate Detection')

elif option == "Video Processing":
    video_file = st.file_uploader("Input a video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        video_bytes = video_file.read()
        video_path = f"temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        process_video(video_path)
        os.remove(video_path)

elif option == "Live Feed Processing":
    process_live_feed()