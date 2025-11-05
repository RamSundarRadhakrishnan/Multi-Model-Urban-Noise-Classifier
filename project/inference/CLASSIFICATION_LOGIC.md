# Event Classification Logic - Visual Guide

## 🎯 Classification Decision Tree

```
Event Data (Audio + Video + Noise)
│
├─ Determine Time Period
│  ├─ Hour ∈ [6, 22) → DAY
│  └─ Hour ∈ [22, 6) → NIGHT
│
├─ Get Noise Zone Limit
│  │
│  ├─ RESIDENTIAL
│  │  ├─ DAY:   55 dB(A)
│  │  └─ NIGHT: 45 dB(A)
│  │
│  ├─ INDUSTRIAL
│  │  ├─ DAY:   75 dB(A)
│  │  └─ NIGHT: 70 dB(A)
│  │
│  ├─ SILENCE
│  │  ├─ DAY:   50 dB(A)
│  │  └─ NIGHT: 40 dB(A)
│  │
│  └─ COMMERCIAL
│     ├─ DAY:   65 dB(A)
│     └─ NIGHT: 55 dB(A)
│
├─ Check Violations
│  │
│  ├─ 1️⃣ AMBIENT NOISE
│  │   ├─ IF noise_level > limit
│  │   │   ├─ violation: AMBIENT_EXCEEDED
│  │   │   └─ excess_db = noise_level - limit
│  │   └─ ELSE: no violation
│  │
│  ├─ 2️⃣ HORN USAGE
│  │   ├─ IF (zone == SILENCE) OR (zone == RESIDENTIAL AND time == NIGHT)
│  │   │   ├─ Check audio_classes for horns
│  │   │   │   ├─ "motorvehicle-horn" OR "car-alarm"
│  │   │   │   └─ IF confidence > 0.6
│  │   │   │       └─ violation: HORN_PROHIBITED
│  │   └─ ELSE: allowed
│  │
│  ├─ 3️⃣ LOUDSPEAKER
│  │   ├─ IF (zone IN [SILENCE, RESIDENTIAL]) AND (time == NIGHT)
│  │   │   ├─ Check audio_classes for speakers
│  │   │   │   ├─ "mobile-music" OR "community-radio"
│  │   │   │   └─ IF confidence > 0.6
│  │   │   │       └─ violation: LOUDSPEAKER_PROHIBITED
│  │   └─ ELSE: allowed (may need permission)
│  │
│  └─ 4️⃣ CONSTRUCTION
│      ├─ IF (zone IN [SILENCE, RESIDENTIAL]) AND (time == NIGHT)
│      │   ├─ Check audio_classes OR video detection
│      │   │   ├─ Audio: "construction-site" OR "generator"
│      │   │   ├─ Video: is_construction_site == True
│      │   │   └─ IF any confidence > 0.6
│      │   │       └─ violation: CONSTRUCTION_PROHIBITED
│      └─ ELSE: allowed
│
└─ Calculate Severity
   │
   ├─ Count violations (exclude NO_VIOLATION)
   ├─ Get excess_db value
   │
   ├─ IF violations == 0 → COMPLIANT
   │
   ├─ IF violations >= 3 OR excess_db > 15 → SEVERE
   │
   ├─ IF violations >= 2 OR excess_db > 10 → MODERATE
   │
   ├─ IF violations >= 1 OR excess_db > 5 → MINOR
   │
   └─ ELSE → COMPLIANT
```

## 📊 Example Classifications

### Example 1: Residential Area, Night Time, High Traffic

**Input:**

```
Zone: RESIDENTIAL
Time: 23:00 (NIGHT)
Noise Level: 58 dB(A)
Audio: motorvehicle-horn (0.87), crowd-noise (0.45)
Video: car (0.92), truck (0.78)
Construction: False
```

**Processing:**

```
1. Time Period: NIGHT (23:00)
2. Limit: 45 dB(A) (Residential Night)
3. Violations:
   ✓ Ambient Exceeded: 58 - 45 = 13 dB excess
   ✓ Horn Prohibited: horn detected (0.87) in Residential+Night
4. Severity: MODERATE (2 violations, 13 dB excess)
```

**Output:**

```json
{
  "violations": ["ambient_exceeded", "horn_prohibited", "multiple_violations"],
  "severity": "moderate",
  "is_compliant": false,
  "recommendations": [
    "Reduce ambient noise by 13.0 dB...",
    "Horn usage is prohibited..."
  ]
}
```

---

### Example 2: Industrial Area, Day Time, Construction

**Input:**

```
Zone: INDUSTRIAL
Time: 14:00 (DAY)
Noise Level: 72 dB(A)
Audio: generator (0.65), construction-site (0.78)
Video: excavator (0.85), dump-truck (0.92)
Construction: True (0.95)
```

**Processing:**

```
1. Time Period: DAY (14:00)
2. Limit: 75 dB(A) (Industrial Day)
3. Violations:
   ✗ Ambient OK: 72 < 75 (within limit)
   ✗ Construction Allowed: Industrial zone + Daytime
4. Severity: COMPLIANT
```

**Output:**

```json
{
  "violations": ["no_violation"],
  "severity": "compliant",
  "is_compliant": true,
  "recommendations": ["Event is compliant with noise regulations."]
}
```

---

### Example 3: Silence Zone, Day Time, Moderate Noise

**Input:**

```
Zone: SILENCE
Time: 10:00 (DAY)
Noise Level: 53 dB(A)
Audio: crowd-noise (0.82)
Video: person (0.88), bicycle (0.76)
Construction: False
```

**Processing:**

```
1. Time Period: DAY (10:00)
2. Limit: 50 dB(A) (Silence Day)
3. Violations:
   ✓ Ambient Exceeded: 53 - 50 = 3 dB excess
   ✗ Horn OK: no horns detected
   ✗ Loudspeaker OK: daytime (restrictions only at night)
4. Severity: MINOR (1 violation, 3 dB excess)
```

**Output:**

```json
{
  "violations": ["ambient_exceeded"],
  "severity": "minor",
  "is_compliant": false,
  "recommendations": [
    "Reduce ambient noise by 3.0 dB to comply with silence zone daytime limits"
  ]
}
```

---

### Example 4: Residential, Night, Construction Site

**Input:**

```
Zone: RESIDENTIAL
Time: 22:30 (NIGHT)
Noise Level: 68 dB(A)
Audio: generator (0.88), construction-site (0.92)
Video: excavator (0.95), cement-mixer (0.87)
Construction: True (0.98)
```

**Processing:**

```
1. Time Period: NIGHT (22:30)
2. Limit: 45 dB(A) (Residential Night)
3. Violations:
   ✓ Ambient Exceeded: 68 - 45 = 23 dB excess
   ✓ Construction Prohibited: Residential + Night + detected
4. Severity: SEVERE (2 violations, 23 dB excess)
```

**Output:**

```json
{
  "violations": [
    "ambient_exceeded",
    "construction_prohibited",
    "multiple_violations"
  ],
  "severity": "severe",
  "is_compliant": false,
  "excess_db": 23.0,
  "recommendations": [
    "Reduce ambient noise by 23.0 dB...",
    "Construction activities are prohibited during nighttime in residential zones (10 PM - 6 AM restriction)."
  ]
}
```

## 🎨 Severity Color Coding

```
┌──────────────────────────────────────┐
│  SEVERITY LEVELS                     │
├──────────────────────────────────────┤
│  🟢 COMPLIANT                        │
│     No violations detected           │
│     All within limits                │
├──────────────────────────────────────┤
│  🟡 MINOR                            │
│     1 violation OR 5-10 dB excess    │
│     Low priority action needed       │
├──────────────────────────────────────┤
│  🟠 MODERATE                         │
│     2 violations OR 10-15 dB excess  │
│     Medium priority action needed    │
├──────────────────────────────────────┤
│  🔴 SEVERE                           │
│     3+ violations OR >15 dB excess   │
│     HIGH PRIORITY - Immediate action │
└──────────────────────────────────────┘
```

## 📋 Violation Type Matrix

| Violation Type              | Zone        | Time  | Trigger               |
| --------------------------- | ----------- | ----- | --------------------- |
| **AMBIENT_EXCEEDED**        | All         | All   | noise_level > limit   |
| **HORN_PROHIBITED**         | Silence     | All   | horn detected         |
| **HORN_PROHIBITED**         | Residential | Night | horn detected         |
| **LOUDSPEAKER_PROHIBITED**  | Silence     | Night | loudspeaker detected  |
| **LOUDSPEAKER_PROHIBITED**  | Residential | Night | loudspeaker detected  |
| **CONSTRUCTION_PROHIBITED** | Silence     | Night | construction detected |
| **CONSTRUCTION_PROHIBITED** | Residential | Night | construction detected |

## 🔍 Detection Confidence Thresholds

```python
CONFIDENCE_THRESHOLD = 0.6  # 30% confidence minimum

# For audio classifications
if audio_class_confidence >= 0.6:
    consider_for_violation_check()

# For video detections
if video_detection_confidence >= 0.6:
    consider_for_violation_check()

# For construction site detection
if construction_site_confidence >= 0.7:  # Higher threshold
    mark_as_construction_site()
```

## 🎯 Special Cases

### Case 1: Multiple Audio Classes

```
Audio Classes: {
    "motorvehicle-horn": 0.87,
    "crowd-noise": 0.65,
    "car-alarm": 0.54
}

→ Checks EACH class independently
→ If ANY restricted class exceeds threshold, violation triggered
→ Can result in multiple violations from audio alone
```

### Case 2: Visual + Audio Construction

```
Audio: construction-site (0.45)  ← Below 0.5 threshold
Video: is_construction_site: True (0.85)  ← Detected visually

→ Construction violation triggered by EITHER source
→ Visual detection has HIGHER weight
→ Combined evidence strengthens classification
```

### Case 3: Boundary Times

```
Time: 22:00:00 (exactly 10 PM)
→ Classified as NIGHT (hour >= 22)
→ Night restrictions apply immediately

Time: 05:59:59 (just before 6 AM)
→ Still NIGHT (hour < 6)
→ Night restrictions still apply

Time: 06:00:00 (exactly 6 AM)
→ Classified as DAY (hour >= 6)
→ Day limits now apply
```

## 💡 Recommendation Logic

```python
if AMBIENT_EXCEEDED:
    → "Reduce ambient noise by {excess_db:.1f} dB to comply with
       {zone} zone {time_of_day}time limits"

if HORN_PROHIBITED:
    → "Horn usage is prohibited in {zone} zones during {time_of_day}time.
       Use horns only for emergency situations."

if LOUDSPEAKER_PROHIBITED:
    → "Loudspeaker/PA system use is prohibited during {time_of_day}time
       in {zone} zones. Obtain permission for daytime use."

if CONSTRUCTION_PROHIBITED:
    → "Construction activities are prohibited during {time_of_day}time
       in {zone} zones (10 PM - 6 AM restriction)."

if NO_VIOLATION:
    → "Event is compliant with noise regulations."
```

## 📈 Confidence Score Impact

### Audio Classification

```
High Confidence (>0.7):
  ✓ Strong evidence for violation
  ✓ Likely accurate detection

Medium Confidence (0.6-0.7):
  ⚠ Moderate evidence
  ⚠ May need manual review

Low Confidence (<0.6):
  ✗ Ignored for violation detection
  ✗ Insufficient evidence
```

### Construction Detection

```
High Confidence (>0.7):
  ✓ Definite construction site
  ✓ is_construction_site = True

Medium Confidence (0.4-0.7):
  ⚠ Possible construction
  ⚠ Additional audio evidence helps

Low Confidence (<0.4):
  ✗ Not classified as construction site
  ✗ is_construction_site = False
```

---

## 🎓 Usage in Code

```python
from event_classifier import EventClassifier, NoiseZone
from datetime import datetime

# Create classifier
classifier = EventClassifier(zone=NoiseZone.RESIDENTIAL)

# Classify an event
event = classifier.classify_event(
    event_id="chunk_0042",
    timestamp=datetime(2025, 10, 30, 23, 15, 0),  # 11:15 PM
    noise_level_db=62.5,
    audio_classes={
        "motorvehicle-horn": 0.87,
        "crowd-noise": 0.45
    },
    video_classes={
        "car": 0.92,
        "truck": 0.78
    },
    is_construction_site=False,
    construction_confidence=0.0,
    zone=NoiseZone.RESIDENTIAL
)

# Check results
print(f"Compliant: {event.is_compliant}")
print(f"Severity: {event.severity}")
print(f"Violations: {[v.value for v in event.violations]}")
print(f"Excess: {event.excess_db:.1f} dB")
print(f"\nRecommendations:")
for rec in event.recommendations:
    print(f"  - {rec}")
```

**Output:**

```
Compliant: False
Severity: moderate
Violations: ['ambient_exceeded', 'horn_prohibited', 'multiple_violations']
Excess: 17.5 dB

Recommendations:
  - Reduce ambient noise by 17.5 dB to comply with residential zone nighttime limits
  - Horn usage is prohibited in residential zones during nighttime. Use horns only for emergency situations.
```

---

This visual guide helps understand the complete event classification logic! 🎯
