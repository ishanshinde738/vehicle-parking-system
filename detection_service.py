"""
Vehicle Detection Service
Handles all vehicle detection, classification, and parking logic
"""

from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path

class VehicleDetectionService:
    """Service for detecting and classifying vehicles"""
    
    # Class mapping from YOLO classes to display categories
    CLASS_MAPPING = {
        'bus': 'Bus',
        'car': 'Car',
        'microbus': 'Bus',
        'motorbike': '2-Wheeler',
        'pickup-van': 'Truck',
        'truck': 'Truck'
    }
    
    # Classes that get parking (only cars)
    PARKING_CLASSES = ['car']
    
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize detection service
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = ['bus', 'car', 'microbus', 'motorbike', 'pickup-van', 'truck']
        
        # Create uploads directory for saving detection images
        self.uploads_dir = Path('uploads')
        self.uploads_dir.mkdir(exist_ok=True)
        
        print(f"🚗 Vehicle Detection Service initialized")
        print(f"   Model: {model_path}")
        print(f"   Confidence threshold: {confidence_threshold}")
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            print(f"\n📥 Loading model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect_vehicles(self, image):
        """
        Detect vehicles in an image
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            list: List of detection dictionaries
        """
        if self.model is None:
            if not self.load_model():
                return []
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            
            # Process detections
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get detection details
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    # Get class name
                    original_class = self.class_names[class_id]
                    display_category = self.CLASS_MAPPING.get(original_class, 'Unknown')
                    
                    # Check if parking applicable
                    parking_applicable = original_class in self.PARKING_CLASSES
                    
                    detection = {
                        'class_id': class_id,
                        'original_class': original_class,
                        'display_category': display_category,
                        'confidence': confidence,
                        'bbox': bbox.tolist(),
                        'parking_applicable': parking_applicable
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def save_detection_image(self, image, detections, prefix='detection'):
        """
        Save image with detection annotations
        
        Args:
            image: OpenCV image
            detections: List of detection dictionaries
            prefix: Filename prefix
            
        Returns:
            str: Saved image path
        """
        try:
            # Create annotated image
            annotated = image.copy()
            
            # Colors for different categories
            colors = {
                'Bus': (0, 255, 255),      # Yellow
                'Car': (0, 255, 0),        # Green
                '2-Wheeler': (255, 0, 0),  # Blue
                'Truck': (0, 165, 255)     # Orange
            }
            
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                category = det['display_category']
                confidence = det['confidence']
                
                # Get color
                color = colors.get(category, (255, 255, 255))
                
                # Draw rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{category} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(annotated, 
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = self.uploads_dir / filename
            
            # Save image
            cv2.imwrite(str(filepath), annotated)
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error saving detection image: {e}")
            return None
    
    def classify_vehicle(self, original_class):
        """
        Map original class to display category
        
        Args:
            original_class: Original detected class name
            
        Returns:
            str: Display category
        """
        return self.CLASS_MAPPING.get(original_class, 'Unknown')
    
    def is_parking_applicable(self, original_class):
        """
        Check if vehicle class gets parking
        
        Args:
            original_class: Original detected class name
            
        Returns:
            bool: True if parking applicable
        """
        return original_class in self.PARKING_CLASSES
    
    def get_detection_summary(self, detections):
        """
        Get summary of detections by category
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            dict: Summary with counts per category
        """
        summary = {
            'total': len(detections),
            'by_category': {},
            'parking_applicable': 0
        }
        
        for det in detections:
            category = det['display_category']
            
            if category not in summary['by_category']:
                summary['by_category'][category] = 0
            
            summary['by_category'][category] += 1
            
            if det['parking_applicable']:
                summary['parking_applicable'] += 1
        
        return summary

def test_detection_service():
    """Test the detection service with a sample image"""
    
    print("=" * 70)
    print("TESTING DETECTION SERVICE")
    print("=" * 70)
    
    # Path to trained model
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print(f"   Please copy best.pt from training/runs/detect/vehicle_detection_v1/weights/")
        return
    
    # Initialize service
    service = VehicleDetectionService(model_path, confidence_threshold=0.5)
    
    if not service.load_model():
        print(f"❌ Failed to load model")
        return
    
    # Test with sample image
    test_image_path = r"C:\Users\admin\OneDrive - Asia Investments Private Limited\Desktop\vehicle_parking_system\vehicle_parking_system\dataset\vehicle-detection.v1i.yolov8-obb\test\images"
    test_image_path = Path(test_image_path)
    
    if not test_image_path.exists():
        print(f"\n⚠️  Test images directory not found")
        return
    
    # Get a test image
    test_images = list(test_image_path.glob('*.jpg'))[:1] + list(test_image_path.glob('*.png'))[:1]
    
    if not test_images:
        print(f"\n⚠️  No test images found")
        return
    
    test_img_path = test_images[0]
    print(f"\n🖼️  Testing with image: {test_img_path.name}")
    
    # Load image
    image = cv2.imread(str(test_img_path))
    
    if image is None:
        print(f"❌ Failed to load image")
        return
    
    # Detect vehicles
    print(f"\n🔍 Running detection...")
    detections = service.detect_vehicles(image)
    
    if len(detections) > 0:
        print(f"\n✅ Detected {len(detections)} vehicle(s):")
        
        for idx, det in enumerate(detections, 1):
            print(f"\n   Vehicle {idx}:")
            print(f"      Original class: {det['original_class']}")
            print(f"      Display category: {det['display_category']}")
            print(f"      Confidence: {det['confidence']:.2%}")
            print(f"      Parking applicable: {'Yes' if det['parking_applicable'] else 'No'}")
        
        # Get summary
        summary = service.get_detection_summary(detections)
        print(f"\n📊 Detection Summary:")
        print(f"   Total vehicles: {summary['total']}")
        print(f"   Parking applicable: {summary['parking_applicable']}")
        print(f"\n   By category:")
        for category, count in summary['by_category'].items():
            print(f"      {category}: {count}")
        
        # Save annotated image
        saved_path = service.save_detection_image(image, detections, prefix='test')
        if saved_path:
            print(f"\n💾 Annotated image saved to: {saved_path}")
            
            # Display image
            annotated = cv2.imread(saved_path)
            cv2.imshow('Detection Result', annotated)
            print(f"\n💡 Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"\n⚠️  No vehicles detected in the image")
    
    print(f"\n✅ Detection service test complete!")

if __name__ == "__main__":
    test_detection_service()