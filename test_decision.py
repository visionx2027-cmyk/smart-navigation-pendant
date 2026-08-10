from decision_engine.decision import get_distance_tier, build_message

for distance in [3.5, 2.8, 1.9, 1.0, 0.7, 0.3]:
    threshold, label, vibration = get_distance_tier(distance)
    message = build_message("person", "ahead", label)
    print(f"Distance {distance}m -> {message} | vibration: {vibration}")