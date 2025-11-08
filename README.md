# 🚗 Counting Total Number of Cars in Video using YOLOv8

## Objective
Detect and count the total number of cars in a given video using a YOLOv8-based object detection model.

##  Workflow
1. Frame Extraction – Extracted frames from the video for dataset creation.
2. Annotation – Labeled cars using Roboflow (YOLO format).
3. Model Training – Trained YOLOv8n model on annotated data (80% train, 20% valid).
4. Evaluation – Calculated mAP, Precision, and Recall.
5. Inference – Used trained model to count cars in each video frame.

##  Results
- mAP50: 0.91
- Precision: 0.89
- Recall: 0.86
- Final output video shows total cars detected per frame.

##  Folder Structure
```
Counting-Total-Number-of-Cars-in-Video/
├── car_counting_workflow.ipynb
├── My_First_Project/
│   ├── data.yaml
│   ├── train/
│   └── valid/
└── README.md
```

##  Run Inference
from ultralytics import YOLO
model = YOLO("model/best.pt")
results = model.predict(source="Video.mp4", save=True, conf=0.5)

## 🔗 Resources
- Output Video: https://drive.google.com/file/d/165UzQzxfA645TI9kMUU6AGAiNaJ8IhyM/view?usp=sharing
- Input Video: https://drive.google.com/file/d/11RtptjgNaydjXoDrR3o2fTuaV7JMim1-/view?usp=sharing

## Author
Darshan S
Data Scientist Trainee Intern
GitHub: https://github.com/Darshanshet23
