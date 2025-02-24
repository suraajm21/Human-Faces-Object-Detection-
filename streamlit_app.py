import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -----------------------
# LOADING DATA
# -----------------------
df_final = pd.read_csv("data/final_annotations.csv")

try:
    yolo_model = YOLO("C:/Users/suraa/OneDrive/Documents/Final Project/runs/detect/face_detection5/weights/best.pt")
except:
    yolo_model = None

# -----------------------
# STREAMLIT LAYOUT
# -----------------------
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox(
    "Menu",
    ["Data", "EDA - Visual", "Prediction"]  # Combining all face detection under 'Prediction'
)

# -----------------------
# SIDEBAR 1: DATA
# -----------------------
if menu == "Data":
    st.title("Data")
    st.write("## 1. Dataset Used for Model Building")
    st.dataframe(df_final.head(20))

    st.write("## 2. Model Performance Metric Dataset")

    if yolo_model is not None:
        with st.spinner("Running model.val() to compute metrics..."):
            metrics = yolo_model.val(data="data.yaml", workers=0)

        if metrics:
            st.write("### YOLO Model Validation Metrics")

            mp, mr, map50, map_ = metrics.box.mean_results()

            st.write(f"**Precision**: {mp:.3f}")
            st.write(f"**Recall**: {mr:.3f}")
        else:
            st.warning("No metrics were returned. Check that your data.yaml is correct.")

# -----------------------
# SIDEBAR 2: EDA - VISUAL
# -----------------------
elif menu == "EDA - Visual":
    st.title("EDA - Visual")

    # Distribution of faces per image
    st.write("### Faces per Image")
    faces_per_image = df_final.groupby('image_name').size().reset_index(name='count')
    fig_faces = px.histogram(
        faces_per_image,
        x='count',
        nbins=10,
        title="Distribution of number of faces per image"
    )
    st.plotly_chart(fig_faces)

    df_final['width'] = df_final['x1'] - df_final['x0']
    df_final['height'] = df_final['y1'] - df_final['y0']

    # Bounding Box Width vs Height
    st.write("### Bounding Box Width vs Height")
    fig_bbox_size = px.scatter(
        df_final,
        x='width',
        y='height',
        title="Bounding Box Width vs Height"
    )
    st.plotly_chart(fig_bbox_size)

# -----------------------
# SIDEBAR 3: PREDICTION
# -----------------------
elif menu == "Prediction":
    st.title("Face Detection / Prediction")

    if not yolo_model:
        st.error("YOLO model not loaded! Please check your model path.")
    else:
        # Creating a sub-menu to switch between Image, Webcam, and Video
        detection_mode = st.radio(
            "Select a Detection Mode:",
            ("Image", "Webcam", "Video")
        )

        # -----------------------
        # 1. IMAGE DETECTION
        # -----------------------
        if detection_mode == "Image":
            st.subheader("Face Detection on a Single Image")
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("Detect Faces"):
                    open_cv_image = np.array(image)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()  # converting RGB->BGR
                    results = yolo_model.predict(open_cv_image)
                    annotated_frame = results[0].plot()

                    st.image(annotated_frame, caption="Detected Faces", use_column_width=True)
            else:
                st.write("Please upload an image first...")

        # -----------------------
        # 2. WEBCAM DETECTION
        # -----------------------
        elif detection_mode == "Webcam":
            st.subheader("Real-time Face Detection via Webcam")
            st.warning(
                "Press 'Start Detection' to open your webcam. "
                "Allow webcam access in your browser. "
                "Close/kill the app or interrupt the process to stop."
            )

            if st.button("Start Detection"):
                cap = cv2.VideoCapture(0)  # 0 = default webcam
                frame_window = st.image([])  

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from camera.")
                        break

                    results = yolo_model.predict(frame)
                    annotated_frame = results[0].plot()

                    # Converting BGR->RGB for Streamlit
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(annotated_frame_rgb)

                cap.release()
                cv2.destroyAllWindows()

        # -----------------------
        # 3. VIDEO DETECTION
        # -----------------------
        elif detection_mode == "Video":
            st.subheader("Face Detection on an Uploaded Video")
            video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
            if video_file is not None:
                # Saving the uploaded video to a temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                tfile_name = tfile.name
                tfile.close()

                # Creating a placeholder to display the frames
                stframe = st.empty()

                # Opening the video file
                cap = cv2.VideoCapture(tfile_name)

                # Reading and displaying each frame
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = yolo_model.predict(frame)
                    annotated_frame = results[0].plot()

                    # Converting BGR to RGB before displaying in Streamlit
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(annotated_frame_rgb)

                cap.release()
                os.remove(tfile_name)
                st.write("Finished processing the video!")
            else:
                st.write("Please upload a video file to begin.")
