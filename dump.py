 spoken_this_cycle = False

            # Priority 1: obstacle danger zone always wins
            if distance_label == "STOP":
                print("Obstacle: STOP")
                spoken_this_cycle = False

            # Priority 2: object detection
            if not spoken_this_cycle:
                #print("ENTERING OBJECT DETECTION")
                object_events = detector.get_object_events()
                #print("YOLO EVENTS:", object_events)

                for event in object_events:
                    label = event["label"]
                    message = f"{label} detected"
                    print(f"OBJECT DETECTED: {message}")

                    try_speak_and_vibrate(
                        f"{label}_detected",
                        message
                    )
                    spoken_this_cycle = True
                    break

            # Priority 3: face recognition announcements
            if not spoken_this_cycle:
                for fe in face_events:

                    if fe["should_announce"]:
                        if fe["name"] == "Unknown":
                            message = f"Unknown person {fe['direction']}"
                        else:
                             else:
                            # VERIFY: this line was truncated in your paste
                            # as 'if fe[">' — confirm this matches your
                            # actual conditional/key against the real file.
                            message = f"{fe['name']} is {fe['direction']}"

                        # VERIFY: this line was truncated in your paste as
                        # "try_speak_and_vibrate(f\"face_{fe['name']}_{fe['>
                        # — confirm the tag string matches your real file.
                        try_speak_and_vibrate(
                            f"face_{fe['name']}_{fe['direction']}",
                            message
                        )
                        spoken_this_cycle = True
                        break
            # ============================================================

            # --- OCR: on-demand only, since it's slow ---
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cam.stop()
        vibration_cleanup()
        sensor_cleanup()