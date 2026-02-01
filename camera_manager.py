"""
Camera Manager
Handles IP camera connections and frame capture
"""

import cv2
import threading
import queue
import time
from datetime import datetime

class CameraManager:
    """Manages camera connection and frame capture"""
    
    def __init__(self, camera_url, camera_id='CAMERA_1'):
        """
        Initialize camera manager
        
        Args:
            camera_url: RTSP URL or camera index (0 for webcam)
            camera_id: Identifier for this camera
        """
        self.camera_url = camera_url
        self.camera_id = camera_id
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.thread = None
        
        print(f"📷 Camera Manager initialized for {camera_id}")
        print(f"   URL: {camera_url}")
    
    def connect(self):
        """Connect to camera"""
        try:
            print(f"\n🔌 Connecting to camera {self.camera_id}...")
            
            # Try to open camera
            if isinstance(self.camera_url, int):
                # Webcam
                self.capture = cv2.VideoCapture(self.camera_url)
            else:
                # IP camera (RTSP)
                self.capture = cv2.VideoCapture(self.camera_url)
            
            # Check if camera opened successfully
            if not self.capture.isOpened():
                print(f"❌ Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            print(f"✅ Camera {self.camera_id} connected successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to camera {self.camera_id}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from camera"""
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=5)
        
        if self.capture:
            self.capture.release()
            print(f"🔌 Camera {self.camera_id} disconnected")
    
    def start_capture(self):
        """Start capturing frames in a separate thread"""
        if not self.capture or not self.capture.isOpened():
            print(f"❌ Camera {self.camera_id} not connected")
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        print(f"▶️  Camera {self.camera_id} capture started")
        return True
    
    def stop_capture(self):
        """Stop capturing frames"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"⏹️  Camera {self.camera_id} capture stopped")
    
    def _capture_loop(self):
        """Capture loop running in separate thread"""
        while self.is_running:
            try:
                ret, frame = self.capture.read()
                
                if not ret:
                    print(f"⚠️  Failed to read frame from {self.camera_id}")
                    time.sleep(1)
                    continue
                
                # Add frame to queue (drop old frames if queue is full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                
                # Small sleep to control frame rate
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error in capture loop for {self.camera_id}: {e}")
                time.sleep(1)
    
    def get_frame(self):
        """
        Get latest frame from queue
        
        Returns:
            numpy.ndarray: Latest frame or None if no frame available
        """
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None
    
    def is_connected(self):
        """Check if camera is connected"""
        return self.capture is not None and self.capture.isOpened()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("CAMERA MANAGER TEST")
    print("=" * 70)
    
    # Test with webcam (index 0)
    print("\n📹 Testing with webcam...")
    
    camera = CameraManager(camera_url=0, camera_id='TEST_WEBCAM')
    
    if camera.connect():
        camera.start_capture()
        
        print("\n📸 Capturing 10 frames...")
        for i in range(10):
            frame = camera.get_frame()
            if frame is not None:
                print(f"   Frame {i+1}: {frame.shape}")
            else:
                print(f"   Frame {i+1}: No frame")
            time.sleep(1)
        
        camera.stop_capture()
        camera.disconnect()
        
        print("\n✅ Camera test completed!")
    else:
        print("\n❌ Camera test failed!")
    
    print("\n" + "=" * 70)
    print("💡 To use with IP camera, change camera_url to RTSP URL:")
    print("   Example: 'rtsp://192.168.1.100:554/stream'")
    print("=" * 70)