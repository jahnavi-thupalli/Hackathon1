import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from transformers import pipeline
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io

# Load models
@st.cache_resource
def load_models():
    yolo_model = YOLO('yolov8n.pt')
    text_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")
    return yolo_model, text_pipe

model, pipe = load_models()

# Utility functions
def describe_position(x_center, frame_width):
    if x_center < frame_width / 3:
        return "on your left"
    elif x_center < 2 * frame_width / 3:
        return "ahead"
    else:
        return "on your right"

def estimate_distance(x1, x2):
    width_pixels = x2 - x1
    return round(2.0 * (1 / (width_pixels / 100)), 1)

def describe_scene_direct(detections, frame_width):
    if not detections:
        return "Nothing detected."

    descriptions = []
    for det in detections:
        label = det["label"]
        x1, _, x2, _ = det["bbox"]
        x_center = (x1 + x2) / 2
        position = describe_position(x_center, frame_width)
        distance = estimate_distance(x1, x2)
        descriptions.append(f"{label} {position}, approximately {distance} meters")

    summary = ", ".join(descriptions)
    prompt = f"The following objects were detected: {summary}."

    response = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)[0]["generated_text"]
    return response.split("</s>")[-1].strip()

def detect_on_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({"label": label, "bbox": [x1, y1, x2, y2]})

        # Draw boxes
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    description = describe_scene_direct(detections, img_rgb.shape[1])
    return img_rgb, description

def speak_text(text, filename="output.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    audio_file = open(filename, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

# Streamlit UI
st.title("🔍 Object Detection + Scene Description (YOLOv8 + TinyLlama)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", channels="BGR")

    if st.button("Detect and Describe"):
        with st.spinner("Processing..."):
            result_img, description = detect_on_image(image)
            st.image(result_img, caption="Detections", channels="RGB")
            st.markdown(f"**Description:** {description}")

            audio = speak_text(description)
            st.audio(audio.export(format="mp3").read(), format="audio/mp3")
