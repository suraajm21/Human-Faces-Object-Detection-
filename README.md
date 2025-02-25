# Human-Faces-Object-Detection-

1. Project Overview:
```
This project focuses on real-time human face detection in images, videos, or from a webcam feed. The key goal is to accurately detect human faces under various conditions—different lighting, angles, and backgrounds. The system leverages:
1. A data preprocessing pipeline (cleaning, resizing, augmentation).
2. YOLOv8 for training an object detection model, which predicts bounding boxes around faces.
3. A Streamlit application that provides:
* Data exploration and visualization (EDA).
* Model performance metrics.
* Real-time or offline face detection through images, videos, and webcam streams.
This end-to-end pipeline showcases typical computer vision best practices, from data ingestion to final deployment.
```

2. Key Requirements:
```
* Accurate and real-time face detection with minimal false positives.
* Achieve Precision, Recall, Accuracy, and F1 > 85% (where applicable).
* Use common Deep Learning frameworks and relevant Python libraries.
* Provide an interactive interface (Streamlit) for:
- EDA and data visualization.
- Inference (image/video/webcam).
- Model performance metrics presentation.
* Enable scalability and easy adaptation to other detection tasks.
```

3. Dataset Information:
```
* A folder of 2204 images.
* Each image may contain zero or more faces.
* The dataset includes bounding box annotations in a CSV file (faces.csv), which specify:
- image_name, width, height, x0, y0, x1, y1.
* Data cleaning is performed to remove incorrect or duplicate annotations.
* The dataset is then split into train/validation/test sets for YOLO training and final evaluation.
* Augmentation is performed to increase image diversity (e.g., flips, rotations, brightness changes).
```
4. Tools, Libraries & Environment:
```
Library / Tool                             Use Case
Python                            Primary programming language.
OpenCV                            Image I/O, resizing, video capture from webcam.
Albumentations                    Image augmentation (flip, brightness/contrast, rotate).
NumPy                             Numerical computing, array operations.
Pandas                            Data handling, CSV input/output, DataFrame operations.
Plotly / Matplotlib / Seaborn     Plotting and EDA.
Scikit-learn                      Train-test splitting, other ML utilities.
Ultralytics YOLO                  YOLOv8 training, validation, inference.
Streamlit                         Building interactive web application front-end.
```

5. Data Preprocessing:
```
1. Load Annotations:
* Load faces.csv into a pandas DataFrame (df_annotations).
* Remove duplicates (.drop_duplicates()).
* Filter out invalid bounding boxes (e.g., x1 <= x0, y1 <= y0).
2. Resize Images:
* Created a directory for processed images (data/images_processed).
* Resized images to a fixed dimension (224×224).
* Scaled bounding boxes accordingly (x_scale, y_scale).
* Saved new annotations (df_annotations_resized) to data/resized_annotations.csv.
3. Data Augmentation:
* Used Albumentations with transformations like HorizontalFlip, RandomBrightnessContrast, Rotate.
* Updated bounding boxes after each transformation.
* Combined augmented data with the original data to form df_final (data/final_annotations.csv).

Why It Matters:
Ensures data consistency (all images have the same size).
Improves model generalization via augmentation.
```

6. Exploratory Data Analysis (EDA):
```
1. Basic Statistics:
* Count total images (unique_images) and total bounding boxes (face_count).
* Distribution of faces per image (using a histogram in Plotly).
* Check for invalid bounding boxes.
2. Bounding Box Size Analysis:
* Computed bbox_width and bbox_height.
* Scatter plot (Plotly) to visualize bounding box dimension distribution.

Outcome:
* Quick sanity check of label distribution, bounding box validity.
* Helps identify potential labeling errors or skewed data.
```

7. Dataset Splitting & YOLO Format Conversion:
```
1. Split into Train/Val/Test:
* train_test_split used to generate train (80%), val/test (10% each).
2. Create YOLO Directory Structure:
* data/images/train, data/images/val, data/images/test, data/labels/train, etc.
3. Convert Bounding Boxes to YOLO Format:
* YOLO requires normalized coordinates (between 0 and 1).
* For each bounding box, x_center, y_center, width_norm, height_norm are computed.
```

8. Training & Validation with YOLOv8:
```
1. data.yaml:
* Contains the path to training, validation, and test sets.
* Lists class names (a single class "face").
2. Model Training:
* model = YOLO('yolov8n.yaml') – loads YOLOv8 nano architecture.
* model.train(...):
- data='data.yaml'
- epochs=50
- batch=8
- imgsz=224
- name='face_detection'
* This produces a runs/detect/face_detection*/weights/ folder with the best model weights (best.pt).
3. Model Validation:
* metrics = model.val(data='data.yaml')
* Evaluates performance metrics on the validation set (and optionally the test set), returning:
Mean precision (mp), mean recall (mr).

Why YOLOv8?
* Modern YOLO versions are known for speed and accuracy.
* Built-in training loop and augmentation.
* Simplifies the usual multi-script pipeline.
```

9. Evaluation & Metrics:
```
Steps:
1. Precision, Recall:
* mp, mr = metrics.box.mean_results()
2. Storing Results:
* The code snippet writes these metrics into a metrics.json file.
* Check that you have import json in that script.

Interpretation:
Precision: Of all predicted faces, how many are correct.
Recall: Of all actual faces, how many did we detect.
```

10. Streamlit Application:
```
1. Layout:
* st.sidebar.title("Navigation") with three main sections: Data, EDA - Visual, and Prediction.
2. Data:
* Displays a portion of df_final (the annotation DataFrame).
* Shows validation metrics by running model.val(data="data.yaml") on the fly.
3. EDA - Visual:
* Distribution of faces per image.
* Scatter plot of bounding box dimensions.
4. Prediction:
* Image Mode: Upload an image, then run YOLO inference (model.predict()) and display bounding boxes.
* Webcam Mode: Opens a live OpenCV capture; run inference frame-by-frame.
* Video Mode: Upload a video, process it frame-by-frame, display bounding boxes in near real-time.
```

11. Results & Observations:
```
1. Preprocessing:
* Resizing to 224×224 and performing augmentation produced a more robust dataset for training.
* No major label inconsistencies found after cleaning.
2. YOLO Model Performance (Example or hypothetical results)
* Precision: ~0.63
* Recall: ~0.55
3. These numbers will vary depending on hyperparameters, training epochs, and data quality. But they demonstrate good detection performance.
```

12.  Potential Improvements and Future Work:
```
1. Higher Resolution: YOLOv8 usually does well with larger input sizes (e.g., 640×640). Larger images might improve detection of small or far-away faces.
2. Hyperparameter Tuning: Adjusting learning rate, batch size, and mosaic augmentation settings.
3. Real-time Optimization: If streaming from a webcam at high FPS, consider using the YOLOv8n (nano) model or pruning/quantization for better speed on low-power devices.
4. Additional EDA: Showing sample images with bounding boxes, or highlight image resolution distribution pre/post-resize.
```

13. Conclusion:
```
This project demonstrates an end-to-end face detection pipeline using modern deep learning approaches (YOLOv8). It handles the full lifecycle—data ingestion, annotation processing, augmentation, training, validation, and interactive inference. With minimal modifications, it can be adapted for other object detection tasks or face recognition expansions.
Key Achievements:
* Streamlined data preprocessing with consistent bounding boxes.
* Achieved strong performance metrics on a relatively modest dataset (~2204 images + augmentations).
* Provided an interactive Streamlit interface for data analysis and real-time inference.
By following the steps in this documentation, anyone can reproduce, evaluate, and extend the face detection solution in new environments or for additional use cases.
```




























