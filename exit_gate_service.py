"""
Exit Gate Service
Monitors exit camera and processes vehicle exits
"""

import cv2
import time
from datetime import datetime
from camera_manager import CameraManager
from detection_service import VehicleDetectionService
from flask import Flask
from database import db
import sys
import os

# Add application path to sys.path
sys.path.insert(0, os.path.dirname(__file__))

# Import app functions
from app import log_vehicle_exit

def run_exit_gate_service():
    """Main exit gate service"""
    
    print("=" * 70)
    print("EXIT GATE SERVICE")
    print("=" * 70)
    
    # Configuration
    CAMERA_URL = 0  # Use 0 for webcam, or RTSP URL for IP camera
    MODEL_PATH = "best.pt"
    DETECTION_COOLDOWN = 5  # Seconds between detections
    
    print(f"\n⚙️  Configuration:")
    print(f"   Camera: {CAMERA_URL}")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Cooldown: {DETECTION_COOLDOWN}s")
    
    # Initialize camera
    camera = CameraManager(CAMERA_URL, 'EXIT_GATE_1')
    
    if not camera.connect():
        print("\n❌ Failed to connect to camera")
        return
    
    # Initialize detection service
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found: {MODEL_PATH}")
        print("   Please copy best.pt from training/runs/detect/vehicle_detection_v1/weights/")
        camera.disconnect()
        return
    
    detection_service = VehicleDetectionService(MODEL_PATH, confidence_threshold=0.5)
    if not detection_service.load_model():
        print("\n❌ Failed to load model")
        camera.disconnect()
        return
    
    # Initialize Flask app for database access
    app = Flask(__name__)
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "parking_system.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    
    # Start camera capture
    camera.start_capture()
    
    print("\n" + "=" * 70)
    print("✅ EXIT GATE SERVICE RUNNING")
    print("=" * 70)
    print("\n📹 Monitoring exit gate...")
    print("⚠️  Press CTRL+C to stop\n")
    
    last_detection_time = 0
    
    try:
        while True:
            # Get frame from camera
            frame = camera.get_frame()
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Check if cooldown period has passed
            current_time = time.time()
            if current_time - last_detection_time < DETECTION_COOLDOWN:
                time.sleep(0.1)
                continue
            
            # Run detection
            detections = detection_service.detect_vehicles(frame)
            
            if len(detections) > 0:
                print(f"\n🚗 {datetime.now().strftime('%H:%M:%S')} - Detected {len(detections)} vehicle(s)")
                
                # Process each detection
                with app.app_context():
                    for detection in detections:
                        print(f"\n   Vehicle Details:")
                        print(f"      Category: {detection['display_category']}")
                        print(f"      Original Class: {detection['original_class']}")
                        print(f"      Confidence: {detection['confidence']:.2%}")
                        print(f"      Parking Applicable: {detection['parking_applicable']}")
                        
                        # Save detection image
                        image_path = detection_service.save_detection_image(
                            frame, [detection], prefix='exit'
                        )
                        print(f"      Image saved: {image_path}")
                        
                        # Log exit to database
                        success, message = log_vehicle_exit(
                            detection, image_path, 'EXIT_GATE_1'
                        )
                        
                        if success:
                            print(f"      ✅ {message}")
                            if detection['parking_applicable']:
                                print(f"      🅿️  Parking slot released")
                        else:
                            print(f"      ❌ {message}")
                
                last_detection_time = current_time
                print(f"\n⏳ Next detection in {DETECTION_COOLDOWN} seconds...")
            
            # Small delay
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping exit gate service...")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        camera.stop_capture()
        camera.disconnect()
        print("✅ Exit gate service stopped")

if __name__ == "__main__":
    run_exit_gate_service()