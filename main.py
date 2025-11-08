import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
from IPython.display import Video, display
import pandas as pd

def extract_frames_for_annotation(video_path, output_dir, frame_interval=30):
    """
    Extract frames from video for annotation
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_path = f"{output_dir}/frame_{saved_count:04d}.jpg"
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {frame_count} total frames")
    return saved_count

# extract_frames_for_annotation("/teamspace/studios/this_studio/Video.mp4","/teamspace/studios/this_studio/")

def verify_dataset_structure(dataset_path):
    """Verify YOLO dataset structure"""
    dataset_path = Path(dataset_path)
    
    required_dirs = [
        'train/images', 'train/labels',
        'valid/images', 'valid/labels'
    ]
    
    print("Dataset Structure Verification:")
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        exists = full_path.exists()
        if exists:
            count = len(list(full_path.glob('*')))
            print(f"✓ {dir_path}: {count} files")
        else:
            print(f"✗ {dir_path}: NOT FOUND")
    
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        print(f"✓ data.yaml found")
    else:
        print(f"✗ data.yaml NOT FOUND")


# verify_dataset_structure('/teamspace/studios/this_studio/My First Project.v2i.yolov8')

def train_vehicle_detector(data_yaml_path, epochs=50, img_size=640):
    """
    Train YOLOv8 model for vehicle detection
    
    Args:
        data_yaml_path: Path to data.yaml file
        epochs: Number of training epochs
        img_size: Input image size
    """
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano version (fastest)
    # Alternative: 'yolov8s.pt' (small), 'yolov8m.pt' (medium)
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        name='vehicle_counter',
        patience=10,  # Early stopping
        save=True,
        plots=True,
        device=0  # Use 0 for GPU, 'cpu' for CPU
    )
    
    return model, results


# model, results = train_vehicle_detector('My First Project.v2i.yolov8/data.yaml', epochs=50)
def evaluate_model(model, data_yaml_path):
    """Evaluate trained model on validation set"""
    metrics = model.val(data=data_yaml_path)
    
    print("\nModel Performance Metrics:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.p:.3f}")
    print(f"Recall: {metrics.box.r:.3f}")
    
    return metrics

# ✅ Load your trained YOLO model first
# model = YOLO("/teamspace/studios/this_studio/runs/detect/vehicle_counter/weights/best.pt")

# # ✅ Now evaluate on your dataset
# evaluate_model(model, "/teamspace/studios/this_studio/My First Project.v2i.yolov8/data.yaml")



class VehicleCounter:
    """Vehicle counting system with tracking"""
    
    def __init__(self, model_path, conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def process_video(self, input_video, output_video):
        """
        Process video and count vehicles
        
        Args:
            input_video: Path to input video
            output_video: Path to save output video
        """
        cap = cv2.VideoCapture(input_video)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Initialize tracker
        byte_tracker = sv.ByteTrack()
        
        # Annotation helpers
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator()
        
        frame_count = 0
        vehicle_counts = []
        tracked_ids = set()
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Update tracker
            detections = byte_tracker.update_with_detections(detections)
            
            # Count unique vehicles
            if detections.tracker_id is not None:
                for tracker_id in detections.tracker_id:
                    tracked_ids.add(tracker_id)
            
            current_count = len(detections)
            total_unique = len(tracked_ids)
            
            # Prepare labels
            labels = [
                f"ID:{tracker_id}" if tracker_id is not None else "New"
                for tracker_id in (detections.tracker_id if detections.tracker_id is not None else [None] * len(detections))
            ]
            
            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
            # Add count text
            cv2.putText(annotated_frame, f"Current Vehicles: {current_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Total Unique: {total_unique}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(annotated_frame)
            
            # Store data
            vehicle_counts.append({
                'frame': frame_count,
                'current_count': current_count,
                'total_unique': total_unique
            })
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        # Save statistics
        df = pd.DataFrame(vehicle_counts)
        df.to_csv('vehicle_count_stats.csv', index=False)
        
        print(f"\nProcessing Complete!")
        print(f"Total unique vehicles tracked: {total_unique}")
        print(f"Average vehicles per frame: {df['current_count'].mean():.1f}")
        print(f"Max vehicles in frame: {df['current_count'].max()}")
        
        return df, total_unique


# Example usage (uncomment to run):
# counter = VehicleCounter('runs/detect/vehicle_counter/weights/best.pt')
# stats_df, total_vehicles = counter.process_video('Video.mp4', 'output_result.mp4')


def visualize_results(stats_df):
    """Create visualizations of counting results"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Vehicles over time
    axes[0].plot(stats_df['frame'], stats_df['current_count'], label='Current Count')
    axes[0].plot(stats_df['frame'], stats_df['total_unique'], label='Cumulative Unique', linestyle='--')
    axes[0].set_xlabel('Frame Number')
    axes[0].set_ylabel('Vehicle Count')
    axes[0].set_title('Vehicle Count Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Distribution
    axes[1].hist(stats_df['current_count'], bins=20, edgecolor='black')
    axes[1].set_xlabel('Number of Vehicles')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Vehicle Counts per Frame')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vehicle_count_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

visualize_results(stats_df)