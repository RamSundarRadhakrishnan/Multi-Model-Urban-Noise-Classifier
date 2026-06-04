"""
Event Classification System based on Indian Noise Pollution Rules
Classifies audio-visual events according to zone-specific noise regulations
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import json


class NoiseZone(Enum):
    """Noise zone types as per Indian Noise Pollution Rules"""
    RESIDENTIAL = "residential"
    INDUSTRIAL = "industrial"
    SILENCE = "silence"
    COMMERCIAL = "commercial"  # Often treated similar to residential


class TimeOfDay(Enum):
    """Time periods for noise regulation"""
    DAY = "day"      # 6:00 AM - 10:00 PM
    NIGHT = "night"  # 10:00 PM - 6:00 AM


@dataclass
class NoiseLimit:
    """Noise limits for a specific zone and time"""
    zone: NoiseZone
    time_of_day: TimeOfDay
    ambient_limit_db: float
    
    
# Standard noise limits as per Indian Noise Pollution (Regulation and Control) Rules, 2000
NOISE_LIMITS = {
    NoiseZone.RESIDENTIAL: {
        TimeOfDay.DAY: 55.0,
        TimeOfDay.NIGHT: 45.0
    },
    NoiseZone.INDUSTRIAL: {
        TimeOfDay.DAY: 75.0,
        TimeOfDay.NIGHT: 70.0
    },
    NoiseZone.SILENCE: {
        TimeOfDay.DAY: 50.0,
        TimeOfDay.NIGHT: 40.0
    },
    NoiseZone.COMMERCIAL: {
        TimeOfDay.DAY: 65.0,
        TimeOfDay.NIGHT: 55.0
    }
}


class ViolationType(Enum):
    """Types of noise violations"""
    AMBIENT_EXCEEDED = "ambient_exceeded"
    HORN_PROHIBITED = "horn_prohibited"
    LOUDSPEAKER_PROHIBITED = "loudspeaker_prohibited"
    CONSTRUCTION_PROHIBITED = "construction_prohibited"
    MULTIPLE_VIOLATIONS = "multiple_violations"
    NO_VIOLATION = "no_violation"


@dataclass
class EventClassification:
    """Classification result for an event"""
    event_id: str
    zone: NoiseZone
    time_of_day: TimeOfDay
    timestamp: datetime
    
    # Noise measurements
    noise_level_db: float
    ambient_limit_db: float
    noise_exceeded: bool
    excess_db: float
    
    # Detected content
    audio_classes: Dict[str, float]  # class_name -> confidence
    video_classes: Dict[str, float]  # class_name -> confidence
    is_construction_site: bool
    construction_confidence: float
    
    # Violation analysis
    violations: List[ViolationType]
    is_compliant: bool
    severity: str  # "compliant", "minor", "moderate", "severe"
    
    # Detailed reasoning
    violation_details: Dict[str, str]
    recommendations: List[str]

    fusion_metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        data = {
            'event_id': self.event_id,
            'zone': self.zone.value,
            'time_of_day': self.time_of_day.value,
            'timestamp': self.timestamp.isoformat(),
            'noise_level_db': round(self.noise_level_db, 2),
            'ambient_limit_db': round(self.ambient_limit_db, 2),
            'noise_exceeded': self.noise_exceeded,
            'excess_db': round(self.excess_db, 2),
            'audio_classes': {k: round(v, 4) for k, v in self.audio_classes.items()},
            'video_classes': {k: round(v, 4) for k, v in self.video_classes.items()},
            'is_construction_site': self.is_construction_site,
            'construction_confidence': round(self.construction_confidence, 4),
            'violations': [v.value for v in self.violations],
            'is_compliant': self.is_compliant,
            'severity': self.severity,
            'violation_details': self.violation_details,
            'recommendations': self.recommendations
        }

        if self.fusion_metadata is not None:
            data['fusion_metadata'] = self.fusion_metadata

        return data


class EventClassifier:
    """
    Classifies events based on audio-visual inference outputs and noise zone regulations
    """
    
    # Restricted activities by zone and time
    HORN_RESTRICTED_ZONES = {NoiseZone.SILENCE, NoiseZone.RESIDENTIAL}
    LOUDSPEAKER_RESTRICTED_ZONES = {NoiseZone.SILENCE, NoiseZone.RESIDENTIAL}
    CONSTRUCTION_RESTRICTED_ZONES = {NoiseZone.SILENCE, NoiseZone.RESIDENTIAL}
    
    # Audio class mappings to violation types
    HORN_CLASSES = {"motorvehicle-horn", "car-alarm"}
    LOUDSPEAKER_CLASSES = {"mobile-music", "community-radio"}
    CONSTRUCTION_CLASSES = {"construction-site", "generator"}
    
    def __init__(self, zone: NoiseZone = NoiseZone.RESIDENTIAL):
        """
        Initialize event classifier
        
        Args:
            zone: Default noise zone for classification
        """
        self.zone = zone
        
    def get_time_of_day(self, timestamp: datetime) -> TimeOfDay:
        """
        Determine if timestamp is day or night
        
        Args:
            timestamp: Event timestamp
            
        Returns:
            TimeOfDay enum (DAY: 6am-10pm, NIGHT: 10pm-6am)
        """
        hour = timestamp.hour
        if 6 <= hour < 22:  # 6 AM to 10 PM
            return TimeOfDay.DAY
        else:
            return TimeOfDay.NIGHT
    
    def get_noise_limit(self, zone: NoiseZone, time_of_day: TimeOfDay) -> float:
        """Get ambient noise limit for zone and time"""
        return NOISE_LIMITS[zone][time_of_day]
    
    def check_ambient_noise_violation(
        self, 
        noise_level: float, 
        zone: NoiseZone, 
        time_of_day: TimeOfDay
    ) -> Tuple[bool, float]:
        """
        Check if ambient noise level exceeds limits
        
        Returns:
            (is_violated, excess_db)
        """
        limit = self.get_noise_limit(zone, time_of_day)
        excess = noise_level - limit
        return (excess > 0, excess)
    
    def check_horn_violation(
        self,
        audio_classes: Dict[str, float],
        zone: NoiseZone,
        time_of_day: TimeOfDay,
        confidence_threshold: float = 0.6
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check for horn usage violations
        
        Returns:
            (is_violated, detected_horn_classes)
        """
        # Horns restricted in silence zones and at night in residential zones
        is_restricted = (
            zone == NoiseZone.SILENCE or
            (zone == NoiseZone.RESIDENTIAL and time_of_day == TimeOfDay.NIGHT)
        )
        
        if not is_restricted:
            return (False, {})
        
        # Check for horn classes
        detected_horns = {
            cls: conf for cls, conf in audio_classes.items()
            if cls in self.HORN_CLASSES and conf >= confidence_threshold
        }
        
        return (len(detected_horns) > 0, detected_horns)
    
    def check_loudspeaker_violation(
        self,
        audio_classes: Dict[str, float],
        zone: NoiseZone,
        time_of_day: TimeOfDay,
        confidence_threshold: float = 0.6
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check for loudspeaker/PA system violations
        
        Returns:
            (is_violated, detected_loudspeaker_classes)
        """
        # Loudspeakers restricted at night in residential/silence zones
        is_restricted = (
            zone in self.LOUDSPEAKER_RESTRICTED_ZONES and
            time_of_day == TimeOfDay.NIGHT
        )
        
        if not is_restricted:
            return (False, {})
        
        # Check for loudspeaker classes
        detected_speakers = {
            cls: conf for cls, conf in audio_classes.items()
            if cls in self.LOUDSPEAKER_CLASSES and conf >= confidence_threshold
        }
        
        return (len(detected_speakers) > 0, detected_speakers)
    
    def check_construction_violation(
        self,
        audio_classes: Dict[str, float],
        is_construction_site: bool,
        zone: NoiseZone,
        time_of_day: TimeOfDay,
        confidence_threshold: float = 0.6
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check for construction equipment violations
        
        Returns:
            (is_violated, detected_construction_classes)
        """
        # Construction restricted at night in residential/silence zones
        is_restricted = (
            zone in self.CONSTRUCTION_RESTRICTED_ZONES and
            time_of_day == TimeOfDay.NIGHT
        )
        
        if not is_restricted:
            return (False, {})
        
        # Check for construction classes in audio or visual detection
        detected_construction = {
            cls: conf for cls, conf in audio_classes.items()
            if cls in self.CONSTRUCTION_CLASSES and conf >= confidence_threshold
        }
        
        # Also consider visual construction site detection
        if is_construction_site:
            detected_construction['visual_construction_site'] = 1.0
        
        return (len(detected_construction) > 0, detected_construction)
    
    def calculate_severity(
        self,
        violations: List[ViolationType],
        excess_db: float
    ) -> str:
        """
        Calculate violation severity
        
        Returns:
            "compliant", "minor", "moderate", or "severe"
        """
        if ViolationType.NO_VIOLATION in violations:
            return "compliant"
        
        # Severity based on excess noise and number of violations
        num_violations = len([v for v in violations if v != ViolationType.NO_VIOLATION])
        
        if num_violations >= 3 or excess_db > 15:
            return "severe"
        elif num_violations >= 2 or excess_db > 10:
            return "moderate"
        elif num_violations >= 1 or excess_db > 5:
            return "minor"
        else:
            return "compliant"
    
    def generate_recommendations(
        self,
        violations: List[ViolationType],
        zone: NoiseZone,
        time_of_day: TimeOfDay,
        excess_db: float
    ) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if ViolationType.AMBIENT_EXCEEDED in violations:
            recommendations.append(
                f"Reduce ambient noise by {abs(excess_db):.1f} dB to comply with "
                f"{zone.value} zone {time_of_day.value}time limits"
            )
        
        if ViolationType.HORN_PROHIBITED in violations:
            recommendations.append(
                f"Horn usage is prohibited in {zone.value} zones during {time_of_day.value}time. "
                "Use horns only for emergency situations."
            )
        
        if ViolationType.LOUDSPEAKER_PROHIBITED in violations:
            recommendations.append(
                f"Loudspeaker/PA system use is prohibited during {time_of_day.value}time "
                f"in {zone.value} zones. Obtain permission for daytime use."
            )
        
        if ViolationType.CONSTRUCTION_PROHIBITED in violations:
            recommendations.append(
                f"Construction activities are prohibited during {time_of_day.value}time "
                f"in {zone.value} zones (10 PM - 6 AM restriction)."
            )
        
        if not recommendations:
            recommendations.append("Event is compliant with noise regulations.")
        
        return recommendations
    
    def classify_event(
        self,
        event_id: str,
        timestamp: datetime,
        noise_level_db: float,
        audio_classes: Dict[str, float],
        video_classes: Dict[str, float],
        is_construction_site: bool,
        construction_confidence: float,
        zone: Optional[NoiseZone] = None,
        confidence_threshold: float = 0.6
    ) -> EventClassification:
        """
        Classify an event based on all available data
        
        Args:
            event_id: Unique event identifier
            timestamp: Event timestamp
            noise_level_db: Measured A-weighted noise level
            audio_classes: Audio classification results {class_name: confidence}
            video_classes: Video detection results {class_name: confidence}
            is_construction_site: Whether construction site was detected
            construction_confidence: Confidence of construction detection
            zone: Noise zone (uses default if not specified)
            confidence_threshold: Minimum confidence for class detection
            
        Returns:
            EventClassification with complete analysis
        """
        zone = zone or self.zone
        time_of_day = self.get_time_of_day(timestamp)
        ambient_limit = self.get_noise_limit(zone, time_of_day)
        
        # Check ambient noise violation
        noise_exceeded, excess_db = self.check_ambient_noise_violation(
            noise_level_db, zone, time_of_day
        )
        
        # Check specific violations
        violations = []
        violation_details = {}
        
        if noise_exceeded:
            violations.append(ViolationType.AMBIENT_EXCEEDED)
            violation_details['ambient_exceeded'] = (
                f"Noise level {noise_level_db:.1f} dB exceeds limit of {ambient_limit:.1f} dB "
                f"by {excess_db:.1f} dB"
            )
        
        # Check horn violations
        horn_violated, detected_horns = self.check_horn_violation(
            audio_classes, zone, time_of_day, confidence_threshold
        )
        if horn_violated:
            violations.append(ViolationType.HORN_PROHIBITED)
            violation_details['horn_prohibited'] = (
                f"Horn detected ({', '.join(detected_horns.keys())}) - "
                f"prohibited in {zone.value} zone during {time_of_day.value}time"
            )
        
        # Check loudspeaker violations
        speaker_violated, detected_speakers = self.check_loudspeaker_violation(
            audio_classes, zone, time_of_day, confidence_threshold
        )
        if speaker_violated:
            violations.append(ViolationType.LOUDSPEAKER_PROHIBITED)
            violation_details['loudspeaker_prohibited'] = (
                f"Loudspeaker/PA system detected ({', '.join(detected_speakers.keys())}) - "
                f"prohibited during {time_of_day.value}time in {zone.value} zone"
            )
        
        # Check construction violations
        construction_violated, detected_construction = self.check_construction_violation(
            audio_classes, is_construction_site, zone, time_of_day, confidence_threshold
        )
        if construction_violated:
            violations.append(ViolationType.CONSTRUCTION_PROHIBITED)
            violation_details['construction_prohibited'] = (
                f"Construction activity detected ({', '.join(detected_construction.keys())}) - "
                f"prohibited during {time_of_day.value}time in {zone.value} zone"
            )
        
        # Determine if multiple violations
        if len(violations) >= 2:
            if ViolationType.MULTIPLE_VIOLATIONS not in violations:
                violations.append(ViolationType.MULTIPLE_VIOLATIONS)
        
        # No violations
        if len(violations) == 0:
            violations.append(ViolationType.NO_VIOLATION)
        
        # Calculate severity
        severity = self.calculate_severity(violations, excess_db)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            violations, zone, time_of_day, excess_db
        )
        
        # Determine compliance
        is_compliant = ViolationType.NO_VIOLATION in violations
        
        return EventClassification(
            event_id=event_id,
            zone=zone,
            time_of_day=time_of_day,
            timestamp=timestamp,
            noise_level_db=noise_level_db,
            ambient_limit_db=ambient_limit,
            noise_exceeded=noise_exceeded,
            excess_db=excess_db,
            audio_classes=audio_classes,
            video_classes=video_classes,
            is_construction_site=is_construction_site,
            construction_confidence=construction_confidence,
            violations=violations,
            is_compliant=is_compliant,
            severity=severity,
            violation_details=violation_details,
            recommendations=recommendations
        )
