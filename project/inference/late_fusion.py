AUDIO_TO_VISUAL_MAP = {
    "crowd-noise": ["person", "worker", "crowd"],
    "generator": ["generator", "machinery", "machine", "truck", "excavator"],
    "motorvehicle-horn": ["car", "truck", "bus", "motorcycle", "vehicle", "auto", "taxi"],
    "mobile-music": ["person", "worker", "crowd"],
    "community-radio": ["person", "worker", "crowd"],
    "construction-site": ["excavator", "truck", "crane", "worker", "machinery", "machine", "construction"],
    "motorvehicle-siren": ["car", "truck", "bus", "vehicle", "ambulance", "police", "fire"],
    "car-alarm": ["car", "vehicle", "truck"]
}


def normalize_text(value):
    if value is None:
        return ""

    return str(value).lower().replace("_", " ").replace("-", " ").strip()


def get_value(obj, key, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def get_detections(video_result):
    if video_result is None:
        return []

    if isinstance(video_result, list):
        return video_result

    candidate_keys = [
        "standard_detections",
        "detections",
        "objects",
        "predictions",
        "results"
    ]

    for key in candidate_keys:
        value = get_value(video_result, key)

        if value is not None:
            return value

    return []


def get_detection_class(detection):
    candidate_keys = [
        "class_name",
        "label",
        "name",
        "class",
        "object_class"
    ]

    for key in candidate_keys:
        value = get_value(detection, key)

        if value is not None:
            return str(value)

    return None


def get_detection_confidence(detection):
    candidate_keys = [
        "confidence",
        "score",
        "conf",
        "probability"
    ]

    for key in candidate_keys:
        value = get_value(detection, key)

        if value is not None:
            return float(value)

    return 0.0


def is_visual_match(detected_class, relevant_classes):
    detected = normalize_text(detected_class)

    for relevant_class in relevant_classes:
        relevant = normalize_text(relevant_class)

        if relevant and relevant in detected:
            return True

    return False


def get_best_visual_match(audio_class, video_result):
    relevant_classes = AUDIO_TO_VISUAL_MAP.get(audio_class, [])
    detections = get_detections(video_result)

    best_class = None
    best_confidence = 0.0

    for detection in detections:
        detected_class = get_detection_class(detection)

        if detected_class is None:
            continue

        confidence = get_detection_confidence(detection)

        if is_visual_match(detected_class, relevant_classes) and confidence > best_confidence:
            best_class = detected_class
            best_confidence = confidence

    return best_class, best_confidence


def compute_fusion_score(audio_confidence, visual_confidence, alpha=0.6):
    audio_confidence = float(audio_confidence)
    visual_confidence = float(visual_confidence)

    if visual_confidence <= 0.0:
        return audio_confidence

    return alpha * audio_confidence + (1.0 - alpha) * visual_confidence


def fuse_prediction(audio_class, audio_confidence, video_result=None, alpha=0.6):
    audio_confidence = float(audio_confidence)

    visual_class, visual_confidence = get_best_visual_match(
        audio_class=audio_class,
        video_result=video_result
    )

    fusion_score = compute_fusion_score(
        audio_confidence=audio_confidence,
        visual_confidence=visual_confidence,
        alpha=alpha
    )

    if visual_confidence > 0.0:
        fusion_mode = "audio_visual_late_fusion"
    else:
        fusion_mode = "audio_only_fallback"

    return {
        "audio_class": audio_class,
        "audio_confidence": audio_confidence,
        "visual_class": visual_class,
        "visual_confidence": visual_confidence,
        "fusion_score": fusion_score,
        "fusion_alpha": alpha,
        "fusion_mode": fusion_mode
    }


def fuse_chunk(chunk, alpha=0.6):
    audio_class = chunk["audio_predicted_class"]
    audio_confidence = chunk["audio_confidence"]
    video_result = chunk.get("video_result")

    return fuse_prediction(
        audio_class=audio_class,
        audio_confidence=audio_confidence,
        video_result=video_result,
        alpha=alpha
    )


def fuse_chunks(chunks, alpha=0.6):
    fused_chunks = []

    for chunk in chunks:
        fusion = fuse_chunk(chunk, alpha=alpha)
        fused_chunk = dict(chunk)
        fused_chunk.update(fusion)
        fused_chunks.append(fused_chunk)

    return fused_chunks