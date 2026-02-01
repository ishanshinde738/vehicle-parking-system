"""
Database Initialization Script
Creates all tables and inserts initial data
"""

from flask import Flask
from database import db, VehicleCategory, ParkingSlot, SystemConfig
import os

def init_database():
    """Initialize database with tables and default data"""
    
    print("=" * 70)
    print("DATABASE INITIALIZATION")
    print("=" * 70)
    
    # Create Flask app
    app = Flask(__name__)
    
    # Database configuration
    basedir = os.path.abspath(os.path.dirname(__file__))
    database_path = os.path.join(basedir, 'parking_system.db')
    
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        # Drop all tables (if reinitializing)
        print("\n⚠️  WARNING: This will DELETE existing database!")
        print(f"   Database: {database_path}")
        print("\n❓ Continue? (y/n): ", end="")
        confirm = input().strip().lower()
        
        if confirm != 'y':
            print("\n❌ Database initialization cancelled")
            return False
        
        print("\n📋 Dropping existing tables...")
        db.drop_all()
        print("✅ Existing tables dropped")
        
        # Create all tables
        print("\n📋 Creating new tables...")
        db.create_all()
        print("✅ Tables created successfully!")
        
        # Insert vehicle categories
        print("\n📋 Inserting vehicle categories...")
        
        categories = [
            {'original_class': 'bus', 'display_category': 'Bus', 'parking_applicable': False},
            {'original_class': 'car', 'display_category': 'Car', 'parking_applicable': True},
            {'original_class': 'microbus', 'display_category': 'Bus', 'parking_applicable': False},
            {'original_class': 'motorbike', 'display_category': '2-Wheeler', 'parking_applicable': False},
            {'original_class': 'pickup-van', 'display_category': 'Truck', 'parking_applicable': False},
            {'original_class': 'truck', 'display_category': 'Truck', 'parking_applicable': False},
        ]
        
        for cat in categories:
            category = VehicleCategory(**cat)
            db.session.add(category)
        
        db.session.commit()
        print("✅ Vehicle categories added:")
        for cat in categories:
            parking_status = "✓ Parking" if cat['parking_applicable'] else "✗ No parking"
            print(f"   {cat['original_class']:12} → {cat['display_category']:10} ({parking_status})")
        
        # Create parking slot configuration
        print("\n📋 Creating parking slot configuration...")
        parking = ParkingSlot(
            total_capacity=100,
            occupied_count=0,
            available_count=100
        )
        db.session.add(parking)
        db.session.commit()
        print(f"✅ Parking configuration created:")
        print(f"   Total capacity: 100 slots (cars only)")
        print(f"   Currently available: 100 slots")
        
        # Insert system configuration
        print("\n📋 Inserting system configuration...")
        
        configs = [
            {'config_key': 'PARKING_CAPACITY', 'config_value': '100', 'description': 'Total parking slots for cars'},
            {'config_key': 'ENTRY_CAMERA_URL', 'config_value': 'rtsp://192.168.1.100:554/stream', 'description': 'Entry gate camera RTSP URL'},
            {'config_key': 'EXIT_CAMERA_URL', 'config_value': 'rtsp://192.168.1.101:554/stream', 'description': 'Exit gate camera RTSP URL'},
            {'config_key': 'DETECTION_CONFIDENCE', 'config_value': '0.5', 'description': 'Minimum confidence threshold for detection'},
            {'config_key': 'DETECTION_COOLDOWN', 'config_value': '5', 'description': 'Cooldown seconds between detections'},
            {'config_key': 'ENABLE_ENTRY_GATE', 'config_value': 'true', 'description': 'Enable entry gate detection'},
            {'config_key': 'ENABLE_EXIT_GATE', 'config_value': 'true', 'description': 'Enable exit gate detection'},
        ]
        
        for conf in configs:
            config = SystemConfig(**conf)
            db.session.add(config)
        
        db.session.commit()
        print("✅ System configuration added:")
        for conf in configs:
            print(f"   {conf['config_key']:25} = {conf['config_value']}")
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ DATABASE INITIALIZATION COMPLETE!")
        print("=" * 70)
        print(f"\n📂 Database created at:")
        print(f"   {database_path}")
        print(f"\n📊 Database contains:")
        print(f"   ✓ 7 tables (vehicle_categories, vehicle_entries, vehicle_exits, etc.)")
        print(f"   ✓ 6 vehicle categories")
        print(f"   ✓ 1 parking configuration (100 slots)")
        print(f"   ✓ 7 system settings")
        print(f"\n🎯 Next Steps:")
        print(f"   1. Database is ready for use")
        print(f"   2. Proceed with Flask application development")
        print(f"   3. Update camera URLs in system_config table as needed")
        
        return True

if __name__ == "__main__":
    success = init_database()
    
    if success:
        print("\n✅ Initialization successful!")
    else:
        print("\n❌ Initialization failed!")