"""
Database Models for Vehicle Parking Management System
Uses SQLAlchemy ORM with SQLite database
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class VehicleCategory(db.Model):
    """Vehicle category mapping table"""
    __tablename__ = 'vehicle_categories'
    
    id = db.Column(db.Integer, primary_key=True)
    original_class = db.Column(db.String(50), unique=True, nullable=False)  # bus, car, microbus, etc.
    display_category = db.Column(db.String(50), nullable=False)  # Bus, Car, 2-Wheeler, Truck
    parking_applicable = db.Column(db.Boolean, default=False)  # Only cars get parking
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<VehicleCategory {self.original_class} -> {self.display_category}>'

class VehicleEntry(db.Model):
    """Vehicle entry log table"""
    __tablename__ = 'vehicle_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    category_id = db.Column(db.Integer, db.ForeignKey('vehicle_categories.id'), nullable=False)
    original_class = db.Column(db.String(50), nullable=False)  # Detected class
    display_category = db.Column(db.String(50), nullable=False)  # Display category
    entry_datetime = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    entry_image_path = db.Column(db.String(500))
    detection_confidence = db.Column(db.Float)
    gate_id = db.Column(db.String(50), default='ENTRY_GATE_1')
    status = db.Column(db.String(20), default='IN')  # IN, OUT
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    category = db.relationship('VehicleCategory', backref='entries')
    exit_log = db.relationship('VehicleExit', backref='entry', uselist=False, cascade='all, delete-orphan')
    parking_allocation = db.relationship('ParkingAllocation', backref='entry', uselist=False, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<VehicleEntry {self.id} - {self.display_category} - {self.status}>'

class VehicleExit(db.Model):
    """Vehicle exit log table"""
    __tablename__ = 'vehicle_exits'
    
    id = db.Column(db.Integer, primary_key=True)
    entry_id = db.Column(db.Integer, db.ForeignKey('vehicle_entries.id'), nullable=False)
    exit_datetime = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    exit_image_path = db.Column(db.String(500))
    duration_minutes = db.Column(db.Integer)
    gate_id = db.Column(db.String(50), default='EXIT_GATE_1')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<VehicleExit {self.id} for Entry {self.entry_id}>'

class ParkingSlot(db.Model):
    """Parking slot management (single row for cars)"""
    __tablename__ = 'parking_slots'
    
    id = db.Column(db.Integer, primary_key=True)
    total_capacity = db.Column(db.Integer, nullable=False, default=100)
    occupied_count = db.Column(db.Integer, nullable=False, default=0)
    available_count = db.Column(db.Integer, nullable=False, default=100)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def allocate_slot(self):
        """Allocate a parking slot"""
        if self.available_count > 0:
            self.occupied_count += 1
            self.available_count -= 1
            self.last_updated = datetime.utcnow()
            return True
        return False
    
    def release_slot(self):
        """Release a parking slot"""
        if self.occupied_count > 0:
            self.occupied_count -= 1
            self.available_count += 1
            self.last_updated = datetime.utcnow()
            return True
        return False
    
    def __repr__(self):
        return f'<ParkingSlot {self.occupied_count}/{self.total_capacity}>'

class ParkingAllocation(db.Model):
    """Track individual car parking allocations"""
    __tablename__ = 'parking_allocations'
    
    id = db.Column(db.Integer, primary_key=True)
    entry_id = db.Column(db.Integer, db.ForeignKey('vehicle_entries.id'), nullable=False)
    allocated_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    released_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='ALLOCATED')  # ALLOCATED, RELEASED
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ParkingAllocation {self.id} - {self.status}>'

class SystemConfig(db.Model):
    """System configuration settings"""
    __tablename__ = 'system_config'
    
    id = db.Column(db.Integer, primary_key=True)
    config_key = db.Column(db.String(100), unique=True, nullable=False)
    config_value = db.Column(db.String(500), nullable=False)
    description = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<SystemConfig {self.config_key}={self.config_value}>'

class DailyStats(db.Model):
    """Pre-calculated daily statistics"""
    __tablename__ = 'daily_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    stat_date = db.Column(db.Date, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    total_entries = db.Column(db.Integer, default=0)
    total_exits = db.Column(db.Integer, default=0)
    currently_in = db.Column(db.Integer, default=0)
    peak_hour = db.Column(db.String(10))
    average_duration_minutes = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Unique constraint on date and category
    __table_args__ = (db.UniqueConstraint('stat_date', 'category', name='_date_category_uc'),)
    
    def __repr__(self):
        return f'<DailyStats {self.stat_date} - {self.category}>'