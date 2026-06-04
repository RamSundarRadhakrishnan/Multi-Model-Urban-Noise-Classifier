"""
Event Storage System for Classified Events
Stores, retrieves, and analyzes classified noise pollution events
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd
from event_classifier import EventClassification, NoiseZone, ViolationType


class EventStorage:
    """
    Manages storage and retrieval of classified events
    """
    
    def __init__(self, storage_dir: str = "./output/events"):
        """
        Initialize event storage
        
        Args:
            storage_dir: Directory to store event data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different output formats
        self.json_dir = self.storage_dir / "json"
        self.csv_dir = self.storage_dir / "csv"
        self.summary_dir = self.storage_dir / "summaries"
        
        for directory in [self.json_dir, self.csv_dir, self.summary_dir]:
            directory.mkdir(exist_ok=True)
        
        self.events: List[EventClassification] = []
    
    def add_event(self, event: EventClassification):
        """Add a classified event to storage"""
        self.events.append(event)
    
    def add_events(self, events: List[EventClassification]):
        """Add multiple events to storage"""
        self.events.extend(events)
    
    def save_json(self, filename: Optional[str] = None) -> Path:
        """
        Save all events to JSON file
        
        Args:
            filename: Custom filename (default: events_<timestamp>.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"events_{timestamp}.json"
        
        filepath = self.json_dir / filename
        
        events_data = {
            'metadata': {
                'total_events': len(self.events),
                'generated_at': datetime.now().isoformat(),
                'compliant_events': sum(1 for e in self.events if e.is_compliant),
                'violation_events': sum(1 for e in self.events if not e.is_compliant)
            },
            'events': [event.to_dict() for event in self.events]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_csv(self, filename: Optional[str] = None) -> Path:
        """
        Save events to CSV file
        
        Args:
            filename: Custom filename (default: events_<timestamp>.csv)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"events_{timestamp}.csv"
        
        filepath = self.csv_dir / filename
        
        # Flatten event data for CSV
        csv_data = []
        fusion = event.fusion_metadata or {}
        for event in self.events:
            # Get top audio and video classes
            top_audio = max(event.audio_classes.items(), key=lambda x: x[1]) if event.audio_classes else ("none", 0.0)
            top_video = max(event.video_classes.items(), key=lambda x: x[1]) if event.video_classes else ("none", 0.0)
            
            csv_data.append({
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'zone': event.zone.value,
                'time_of_day': event.time_of_day.value,
                'noise_level_db': round(event.noise_level_db, 2),
                'ambient_limit_db': round(event.ambient_limit_db, 2),
                'excess_db': round(event.excess_db, 2),
                'noise_exceeded': event.noise_exceeded,
                'top_audio_class': top_audio[0],
                'top_audio_confidence': round(top_audio[1], 4),
                'top_video_class': top_video[0],
                'top_video_confidence': round(top_video[1], 4),
                'is_construction_site': event.is_construction_site,
                'construction_confidence': round(event.construction_confidence, 4),
                'violations': '; '.join([v.value for v in event.violations]),
                'is_compliant': event.is_compliant,
                'severity': event.severity,
                'num_violations': len([v for v in event.violations if v != ViolationType.NO_VIOLATION]),
                'fusion_audio_class': fusion.get('audio_class'),
                'fusion_audio_confidence': round(fusion.get('audio_confidence', 0.0), 4),
                'fusion_visual_class': fusion.get('visual_class'),
                'fusion_visual_confidence': round(fusion.get('visual_confidence', 0.0), 4),
                'fusion_score': round(fusion.get('fusion_score', 0.0), 4),
                'fusion_alpha': fusion.get('fusion_alpha'),
                'fusion_mode': fusion.get('fusion_mode'),
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return filepath
    
    def save_detailed_csv(self, filename: Optional[str] = None) -> Path:
        """
        Save detailed events with all audio/video classes to CSV
        
        Args:
            filename: Custom filename (default: events_detailed_<timestamp>.csv)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"events_detailed_{timestamp}.csv"
        
        filepath = self.csv_dir / filename
        
        # Flatten event data with all classes
        csv_data = []
        fusion = event.fusion_metadata or {}
        for event in self.events:
            base_data = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'zone': event.zone.value,
                'time_of_day': event.time_of_day.value,
                'noise_level_db': round(event.noise_level_db, 2),
                'ambient_limit_db': round(event.ambient_limit_db, 2),
                'excess_db': round(event.excess_db, 2),
                'noise_exceeded': event.noise_exceeded,
                'is_construction_site': event.is_construction_site,
                'construction_confidence': round(event.construction_confidence, 4),
                'violations': '; '.join([v.value for v in event.violations]),
                'is_compliant': event.is_compliant,
                'severity': event.severity,
                'fusion_audio_class': fusion.get('audio_class'),
                'fusion_audio_confidence': round(fusion.get('audio_confidence', 0.0), 4),
                'fusion_visual_class': fusion.get('visual_class'),
                'fusion_visual_confidence': round(fusion.get('visual_confidence', 0.0), 4),
                'fusion_score': round(fusion.get('fusion_score', 0.0), 4),
                'fusion_alpha': fusion.get('fusion_alpha'),
                'fusion_mode': fusion.get('fusion_mode'),
            }
            
            # Add all audio classes as columns
            for cls, conf in event.audio_classes.items():
                base_data[f'audio_{cls}'] = round(conf, 4)
            
            # Add all video classes as columns
            for cls, conf in event.video_classes.items():
                base_data[f'video_{cls}'] = round(conf, 4)
            
            csv_data.append(base_data)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return filepath
    
    def save_violations_only(self, filename: Optional[str] = None) -> Path:
        """
        Save only events with violations
        
        Args:
            filename: Custom filename (default: violations_<timestamp>.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violations_{timestamp}.json"
        
        filepath = self.json_dir / filename
        
        violations = [e for e in self.events if not e.is_compliant]
        
        violations_data = {
            'metadata': {
                'total_violations': len(violations),
                'generated_at': datetime.now().isoformat(),
                'severity_breakdown': self.get_severity_breakdown(violations)
            },
            'violations': [event.to_dict() for event in violations]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(violations_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_summary_report(self, filename: Optional[str] = None) -> Path:
        """
        Generate comprehensive summary report
        
        Args:
            filename: Custom filename (default: summary_<timestamp>.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{timestamp}.json"
        
        filepath = self.summary_dir / filename
        
        total_events = len(self.events)
        compliant = sum(1 for e in self.events if e.is_compliant)
        violations = total_events - compliant
        
        # Violation type breakdown
        violation_counts = {}
        for event in self.events:
            for violation in event.violations:
                if violation != ViolationType.NO_VIOLATION:
                    violation_counts[violation.value] = violation_counts.get(violation.value, 0) + 1
        
        # Zone breakdown
        zone_breakdown = {}
        for event in self.events:
            zone = event.zone.value
            if zone not in zone_breakdown:
                zone_breakdown[zone] = {'total': 0, 'compliant': 0, 'violations': 0}
            zone_breakdown[zone]['total'] += 1
            if event.is_compliant:
                zone_breakdown[zone]['compliant'] += 1
            else:
                zone_breakdown[zone]['violations'] += 1
        
        # Time of day breakdown
        time_breakdown = {
            'day': {'total': 0, 'compliant': 0, 'violations': 0},
            'night': {'total': 0, 'compliant': 0, 'violations': 0}
        }
        for event in self.events:
            tod = event.time_of_day.value
            time_breakdown[tod]['total'] += 1
            if event.is_compliant:
                time_breakdown[tod]['compliant'] += 1
            else:
                time_breakdown[tod]['violations'] += 1
        
        # Severity breakdown
        severity_breakdown = self.get_severity_breakdown(self.events)
        
        # Average noise levels
        avg_noise = sum(e.noise_level_db for e in self.events) / total_events if total_events > 0 else 0
        max_noise = max((e.noise_level_db for e in self.events), default=0)
        max_excess = max((e.excess_db for e in self.events if e.noise_exceeded), default=0)
        
        # Construction site statistics
        construction_events = sum(1 for e in self.events if e.is_construction_site)
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'overview': {
                'total_events': total_events,
                'compliant_events': compliant,
                'violation_events': violations,
                'compliance_rate': round(compliant / total_events * 100, 2) if total_events > 0 else 0
            },
            'violation_types': violation_counts,
            'zone_breakdown': zone_breakdown,
            'time_breakdown': time_breakdown,
            'severity_breakdown': severity_breakdown,
            'noise_statistics': {
                'average_noise_db': round(avg_noise, 2),
                'maximum_noise_db': round(max_noise, 2),
                'maximum_excess_db': round(max_excess, 2)
            },
            'construction_statistics': {
                'total_construction_events': construction_events,
                'construction_percentage': round(construction_events / total_events * 100, 2) if total_events > 0 else 0
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def get_severity_breakdown(self, events: List[EventClassification]) -> Dict[str, int]:
        """Get breakdown of events by severity"""
        breakdown = {
            'compliant': 0,
            'minor': 0,
            'moderate': 0,
            'severe': 0
        }
        
        for event in events:
            breakdown[event.severity] = breakdown.get(event.severity, 0) + 1
        
        return breakdown
    
    def filter_by_zone(self, zone: NoiseZone) -> List[EventClassification]:
        """Filter events by zone"""
        return [e for e in self.events if e.zone == zone]
    
    def filter_by_severity(self, severity: str) -> List[EventClassification]:
        """Filter events by severity"""
        return [e for e in self.events if e.severity == severity]
    
    def filter_by_time(self, start: datetime, end: datetime) -> List[EventClassification]:
        """Filter events by time range"""
        return [e for e in self.events if start <= e.timestamp <= end]
    
    def get_violations_only(self) -> List[EventClassification]:
        """Get only events with violations"""
        return [e for e in self.events if not e.is_compliant]
    
    def get_compliant_only(self) -> List[EventClassification]:
        """Get only compliant events"""
        return [e for e in self.events if e.is_compliant]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all events to pandas DataFrame"""
        data = []
        for event in self.events:
            event_dict = event.to_dict()
            # Flatten nested structures for DataFrame
            event_dict['violations'] = '; '.join(event_dict['violations'])
            event_dict['audio_classes'] = str(event_dict['audio_classes'])
            event_dict['video_classes'] = str(event_dict['video_classes'])
            event_dict['violation_details'] = str(event_dict['violation_details'])
            event_dict['recommendations'] = '; '.join(event_dict['recommendations'])
            data.append(event_dict)
        
        return pd.DataFrame(data)
    
    def clear(self):
        """Clear all stored events"""
        self.events = []
    
    def __len__(self) -> int:
        """Return number of stored events"""
        return len(self.events)
    
    def __repr__(self) -> str:
        compliant = sum(1 for e in self.events if e.is_compliant)
        violations = len(self.events) - compliant
        return (f"EventStorage(total={len(self.events)}, "
                f"compliant={compliant}, violations={violations})")
