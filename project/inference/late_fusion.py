AUDIO_TO_VISUAL_MAP = {
    "crowd-noise": ["person", "worker"],
    "generator": ["generator", "machinery", "truck", "excavator"],
    "motorvehicle-horn": ["car", "truck", "bus", "motorcycle", "vehicle"],
    "mobile-music": ["person", "worker"],
    "community-radio": ["person", "worker"],
    "construction-site": ["excavator", "truck", "crane", "worker", "machinery"],
    "motorvehicle-siren": ["car", "truck", "bus", "vehicle"],
    "car-alarm": ["car", "vehicle"]
}


def get_best_visual_match(audio_class, video_result):
    if video_result is None:
        return None, 0.0

    relevant_classes = AUDIO_TO_VISUAL_MAP.get(audio_class, [])
    best_class = None
    best_confidence = 0.0

    detections = getattr(video_result, "standard_detections", [])

    for detection in detections:
        detected_class = detection.class_name.lower()
        confidence = float(detection.confidence)

        for relevant in relevant_classes:
            if relevant in detected_class:
                if confidence > best_confidence:
                    best_class = detected_class
                    best_confidence = confidence

    return best_class, best_confidence


def compute_fusion_score(audio_confidence, visual_confidence, alpha=0.6):
    return alpha * float(audio_confidence) + (1 - alpha) * float(visual_confidence)


def fuse_chunk(chunk, alpha=0.6):
    audio_class = chunk["audio_predicted_class"]
    audio_confidence = float(chunk["audio_confidence"])

    visual_class, visual_confidence = get_best_visual_match(
        audio_class,
        chunk.get("video_result")
    )

    fusion_score = compute_fusion_score(
        audio_confidence,
        visual_confidence,
        alpha=alpha
    )

    return {
        "audio_class": audio_class,
        "audio_confidence": audio_confidence,
        "visual_class": visual_class,
        "visual_confidence": visual_confidence,
        "fusion_score": fusion_score
    }