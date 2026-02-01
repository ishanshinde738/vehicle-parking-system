"""
Vehicle Parking Management System - Main Flask Application
Complete system with Vehicle Counting using Line Crossing Detection
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for, Response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from database import db, VehicleCategory, VehicleEntry, VehicleExit, ParkingSlot, ParkingAllocation, SystemConfig, DailyStats
from detection_service import VehicleDetectionService
from vehicle_counter import VehicleCounter
from datetime import datetime, timedelta, date
from sqlalchemy import func, and_
import os
from pathlib import Path
import cv2
import threading
import time
import uuid

app = Flask(__name__)

# Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "parking_system.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'vehicle-parking-secret-key-2024'
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

# Initialize database
db.init_app(app)

# Ensure upload folder exists
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Global variables
camera_feed = None
camera_lock = threading.Lock()
detection_service_global = None
vehicle_counter_global = None

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

class CameraFeed:
    """Camera feed handler with counting"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.camera = None
        self.is_running = False
        self.frame = None
        
    def start(self):
        """Start camera capture"""
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(self.camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        if self.camera.isOpened():
            self.is_running = True
            print("📹 Camera started successfully")
            return True
        else:
            print("❌ Failed to start camera")
            return False
    
    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        print("📹 Camera stopped")
    
    def get_frame(self):
        """Get current frame from camera"""
        if self.camera and self.camera.isOpened():
            success, frame = self.camera.read()
            if success:
                self.frame = frame
                return frame
        return None
    
    def get_frame_with_counting(self, detection_service, vehicle_counter):
        """Get frame with detection and counting overlay"""
        frame = self.get_frame()
        if frame is None:
            return None
        
        try:
            # Run detection
            detections = detection_service.detect_vehicles(frame)
            
            # Update vehicle counter with detections
            vehicle_counter.update(detections, frame.shape)
            
            # Draw counting visualization
            frame = vehicle_counter.draw_on_frame(frame)
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"❌ Detection/Counting error: {e}")
            return frame

def get_camera_feed():
    """Get or create camera feed instance"""
    global camera_feed
    if camera_feed is None:
        camera_feed = CameraFeed(camera_id=0)
        camera_feed.start()
    return camera_feed

def get_detection_service():
    """Get or create detection service instance"""
    global detection_service_global
    if detection_service_global is None:
        model_path = os.path.join(basedir, 'best.pt')
        if os.path.exists(model_path):
            detection_service_global = VehicleDetectionService(model_path, confidence_threshold=0.5)
            detection_service_global.load_model()
            print("✅ Detection service loaded")
        else:
            print(f"⚠️  Warning: Model not found at {model_path}")
    return detection_service_global

def get_vehicle_counter():
    """Get or create vehicle counter instance"""
    global vehicle_counter_global
    if vehicle_counter_global is None:
        # Line at center (0.5), LEFT=OUT, RIGHT=IN
        vehicle_counter_global = VehicleCounter(
            line_position=0.5,
            direction_mapping={'LEFT': 'OUT', 'RIGHT': 'IN'}
        )
        print("✅ Vehicle counter initialized")
    return vehicle_counter_global

def generate_frames():
    """Generate frames for video streaming with counting"""
    camera = get_camera_feed()
    detection_service = get_detection_service()
    vehicle_counter = get_vehicle_counter()
    
    while True:
        try:
            if detection_service and vehicle_counter:
                frame = camera.get_frame_with_counting(detection_service, vehicle_counter)
            else:
                frame = camera.get_frame()
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # ~30 FPS
            
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
            time.sleep(0.1)

# ==================== WEB ROUTES ====================

@app.route('/')
def dashboard():
    """Main dashboard"""
    parking = ParkingSlot.query.first()
    today = date.today()
    
    # Count entries by category for today
    entries_today = db.session.query(
        VehicleEntry.display_category,
        func.count(VehicleEntry.id).label('count')
    ).filter(
        func.date(VehicleEntry.entry_datetime) == today
    ).group_by(VehicleEntry.display_category).all()
    
    entries_by_category = {cat: count for cat, count in entries_today}
    
    # Count currently IN vehicles by category
    currently_in = db.session.query(
        VehicleEntry.display_category,
        func.count(VehicleEntry.id).label('count')
    ).filter(
        VehicleEntry.status == 'IN'
    ).group_by(VehicleEntry.display_category).all()
    
    currently_in_by_category = {cat: count for cat, count in currently_in}
    
    # Get recent entries
    recent_entries = VehicleEntry.query.order_by(
        VehicleEntry.entry_datetime.desc()
    ).limit(10).all()
    
    return render_template('dashboard.html',
                         parking=parking,
                         entries_by_category=entries_by_category,
                         currently_in_by_category=currently_in_by_category,
                         recent_entries=recent_entries,
                         today=today)

@app.route('/video_feed')
def video_feed():
    """Video streaming route for live feed with counting"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/entries')
def entries():
    """Vehicle entries log"""
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    category_filter = request.args.get('category', '')
    status_filter = request.args.get('status', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    query = VehicleEntry.query
    
    if category_filter:
        query = query.filter(VehicleEntry.display_category == category_filter)
    
    if status_filter:
        query = query.filter(VehicleEntry.status == status_filter)
    
    if date_from:
        date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
        query = query.filter(VehicleEntry.entry_datetime >= date_from_obj)
    
    if date_to:
        date_to_obj = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
        query = query.filter(VehicleEntry.entry_datetime < date_to_obj)
    
    pagination = query.order_by(VehicleEntry.entry_datetime.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    categories = db.session.query(VehicleEntry.display_category).distinct().all()
    categories = [c[0] for c in categories]
    
    return render_template('entries.html',
                         entries=pagination.items,
                         pagination=pagination,
                         categories=categories,
                         filters={
                             'category': category_filter,
                             'status': status_filter,
                             'date_from': date_from,
                             'date_to': date_to
                         })

@app.route('/exits')
def exits():
    """Vehicle exits log"""
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    category_filter = request.args.get('category', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    query = db.session.query(VehicleExit, VehicleEntry).join(
        VehicleEntry, VehicleExit.entry_id == VehicleEntry.id
    )
    
    if category_filter:
        query = query.filter(VehicleEntry.display_category == category_filter)
    
    if date_from:
        date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
        query = query.filter(VehicleExit.exit_datetime >= date_from_obj)
    
    if date_to:
        date_to_obj = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
        query = query.filter(VehicleExit.exit_datetime < date_to_obj)
    
    pagination = query.order_by(VehicleExit.exit_datetime.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    categories = db.session.query(VehicleEntry.display_category).distinct().all()
    categories = [c[0] for c in categories]
    
    return render_template('exits.html',
                         exits=pagination.items,
                         pagination=pagination,
                         categories=categories,
                         filters={
                             'category': category_filter,
                             'date_from': date_from,
                             'date_to': date_to
                         })

@app.route('/parking')
def parking():
    """Parking status and management"""
    parking_slot = ParkingSlot.query.first()
    
    active_allocations = db.session.query(ParkingAllocation, VehicleEntry).join(
        VehicleEntry, ParkingAllocation.entry_id == VehicleEntry.id
    ).filter(
        ParkingAllocation.status == 'ALLOCATED'
    ).order_by(ParkingAllocation.allocated_at.desc()).all()
    
    today = date.today()
    history_today = db.session.query(ParkingAllocation, VehicleEntry).join(
        VehicleEntry, ParkingAllocation.entry_id == VehicleEntry.id
    ).filter(
        func.date(ParkingAllocation.allocated_at) == today
    ).order_by(ParkingAllocation.allocated_at.desc()).limit(50).all()
    
    return render_template('parking.html',
                         parking=parking_slot,
                         active_allocations=active_allocations,
                         history=history_today)

@app.route('/analytics')
def analytics():
    """Analytics and reports"""
    days = request.args.get('days', 7, type=int)
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    daily_entries = db.session.query(
        func.date(VehicleEntry.entry_datetime).label('date'),
        VehicleEntry.display_category,
        func.count(VehicleEntry.id).label('count')
    ).filter(
        func.date(VehicleEntry.entry_datetime) >= start_date
    ).group_by(
        func.date(VehicleEntry.entry_datetime),
        VehicleEntry.display_category
    ).all()
    
    category_distribution = db.session.query(
        VehicleEntry.display_category,
        func.count(VehicleEntry.id).label('count')
    ).filter(
        func.date(VehicleEntry.entry_datetime) >= start_date
    ).group_by(VehicleEntry.display_category).all()
    
    peak_hours = db.session.query(
        func.strftime('%H', VehicleEntry.entry_datetime).label('hour'),
        func.count(VehicleEntry.id).label('count')
    ).filter(
        func.date(VehicleEntry.entry_datetime) >= start_date
    ).group_by('hour').all()
    
    avg_duration = db.session.query(
        VehicleExit.duration_minutes
    ).filter(
        func.date(VehicleExit.exit_datetime) >= start_date,
        VehicleExit.duration_minutes.isnot(None)
    ).all()
    
    avg_duration_minutes = sum([d[0] for d in avg_duration]) / len(avg_duration) if avg_duration else 0
    
    return render_template('analytics.html',
                         daily_entries=daily_entries,
                         category_distribution=category_distribution,
                         peak_hours=peak_hours,
                         avg_duration_minutes=avg_duration_minutes,
                         days=days,
                         start_date=start_date,
                         end_date=end_date)

@app.route('/settings')
def settings():
    """System settings"""
    parking = ParkingSlot.query.first()
    configs = SystemConfig.query.all()
    
    return render_template('settings.html',
                         parking=parking,
                         configs=configs)

@app.route('/settings/update_parking', methods=['POST'])
def update_parking_capacity():
    """Update parking capacity"""
    new_capacity = request.form.get('capacity', type=int)
    
    if new_capacity and new_capacity > 0:
        parking = ParkingSlot.query.first()
        if parking:
            old_capacity = parking.total_capacity
            parking.total_capacity = new_capacity
            parking.available_count = new_capacity - parking.occupied_count
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Parking capacity updated from {old_capacity} to {new_capacity}'
            })
    
    return jsonify({
        'success': False,
        'message': 'Invalid capacity value'
    }), 400

# ==================== API ENDPOINTS ====================

@app.route('/api/vehicle-counts')
def api_vehicle_counts():
    """Get current vehicle counts from counter"""
    vehicle_counter = get_vehicle_counter()
    
    if vehicle_counter:
        counts = vehicle_counter.get_counts()
        
        # Get parking info
        parking = ParkingSlot.query.first()
        total_capacity = parking.total_capacity if parking else 100
        
        parking_info = vehicle_counter.get_parking_availability(total_capacity)
        
        return jsonify({
            'success': True,
            'counts': counts,
            'parking': parking_info
        })
    
    return jsonify({'success': False, 'message': 'Counter not available'}), 500

@app.route('/api/reset-counts', methods=['POST'])
def api_reset_counts():
    """Reset vehicle counts"""
    vehicle_counter = get_vehicle_counter()
    
    if vehicle_counter:
        vehicle_counter.reset_counts()
        return jsonify({'success': True, 'message': 'Counts reset successfully'})
    
    return jsonify({'success': False, 'message': 'Counter not available'}), 500

@app.route('/api/detect-image', methods=['POST'])
def detect_image():
    """Detect vehicles in uploaded image"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'}), 400
    
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        try:
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            detection_service = get_detection_service()
            if not detection_service:
                return jsonify({'success': False, 'message': 'Detection service not available'}), 500
            
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'success': False, 'message': 'Failed to read image'}), 400
            
            detections = detection_service.detect_vehicles(image)
            
            if len(detections) > 0:
                colors = {
                    'Bus': (0, 255, 255),
                    'Car': (0, 255, 0),
                    '2-Wheeler': (255, 0, 0),
                    'Truck': (0, 165, 255)
                }
                
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    category = det['display_category']
                    confidence = det['confidence']
                    color = colors.get(category, (255, 255, 255))
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    label = f"{category} {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            annotated_filename = f"annotated_{unique_filename}"
            annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
            cv2.imwrite(annotated_filepath, image)
            
            return jsonify({
                'success': True,
                'detections': detections,
                'total_detections': len(detections),
                'annotated_image_url': f'/uploads/{annotated_filename}'
            })
            
        except Exception as e:
            print(f"Error detecting image: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500
    
    return jsonify({'success': False, 'message': 'Invalid file type'}), 400

@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    """Detect vehicles in uploaded video"""
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No video provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No video selected'}), 400
    
    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        try:
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            detection_service = get_detection_service()
            if not detection_service:
                return jsonify({'success': False, 'message': 'Detection service not available'}), 500
            
            cap = cv2.VideoCapture(filepath)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            annotated_filename = f"annotated_{unique_filename}"
            annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(annotated_filepath, fourcc, fps, (width, height))
            
            all_detections = []
            frames_processed = 0
            frame_skip = 5
            
            colors = {
                'Bus': (0, 255, 255),
                'Car': (0, 255, 0),
                '2-Wheeler': (255, 0, 0),
                'Truck': (0, 165, 255)
            }
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames_processed += 1
                
                if frames_processed % frame_skip == 0:
                    detections = detection_service.detect_vehicles(frame)
                    all_detections.extend(detections)
                    
                    for det in detections:
                        bbox = det['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        category = det['display_category']
                        confidence = det['confidence']
                        color = colors.get(category, (255, 255, 255))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{category} {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                out.write(frame)
            
            cap.release()
            out.release()
            
            return jsonify({
                'success': True,
                'detections': all_detections,
                'total_detections': len(all_detections),
                'frames_processed': frames_processed,
                'annotated_video_url': f'/uploads/{annotated_filename}'
            })
            
        except Exception as e:
            print(f"Error detecting video: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500
    
    return jsonify({'success': False, 'message': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/parking-status')
def api_parking_status():
    """Get current parking status"""
    parking = ParkingSlot.query.first()
    
    if parking:
        return jsonify({
            'total_capacity': parking.total_capacity,
            'occupied_count': parking.occupied_count,
            'available_count': parking.available_count,
            'last_updated': parking.last_updated.isoformat() if parking.last_updated else None
        })
    
    return jsonify({'error': 'Parking data not found'}), 404

@app.route('/api/latest-entries')
def api_latest_entries():
    """Get latest entries"""
    limit = request.args.get('limit', 10, type=int)
    
    entries = VehicleEntry.query.order_by(
        VehicleEntry.entry_datetime.desc()
    ).limit(limit).all()
    
    return jsonify([{
        'id': entry.id,
        'category': entry.display_category,
        'entry_time': entry.entry_datetime.isoformat(),
        'status': entry.status,
        'confidence': entry.detection_confidence
    } for entry in entries])

@app.route('/api/stats')
def api_stats():
    """Get dashboard statistics"""
    today = date.today()
    
    total_entries_today = VehicleEntry.query.filter(
        func.date(VehicleEntry.entry_datetime) == today
    ).count()
    
    total_exits_today = VehicleExit.query.filter(
        func.date(VehicleExit.exit_datetime) == today
    ).count()
    
    currently_in = VehicleEntry.query.filter(
        VehicleEntry.status == 'IN'
    ).count()
    
    parking = ParkingSlot.query.first()
    
    return jsonify({
        'today': {
            'total_entries': total_entries_today,
            'total_exits': total_exits_today,
            'currently_in': currently_in
        },
        'parking': {
            'total_capacity': parking.total_capacity if parking else 0,
            'occupied_count': parking.occupied_count if parking else 0,
            'available_count': parking.available_count if parking else 0
        }
    })

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("=" * 70)
    print("VEHICLE PARKING MANAGEMENT SYSTEM WITH COUNTING")
    print("=" * 70)
    print("\n🚀 Starting Flask application...")
    print(f"📂 Database: {os.path.join(basedir, 'parking_system.db')}")
    print(f"📂 Uploads: {app.config['UPLOAD_FOLDER']}")
    print(f"\n🌐 Application will run on:")
    print(f"   http://localhost:5000")
    print(f"\n✨ Features Available:")
    print(f"   📸 Image Upload Detection")
    print(f"   🎬 Video Upload Detection")
    print(f"   📹 Live Camera Feed with Vehicle Counting")
    print(f"   🚦 IN/OUT Counting with Line Crossing Detection")
    print(f"   🅿️  Real-time Parking Availability")
    print("\n⚠️  Press CTRL+C to stop the server")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)