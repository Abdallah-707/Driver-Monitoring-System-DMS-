import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import threading
import time
import os
from playsound import playsound  # type: ignore

# ============================================
# Configuration Constants
# ============================================

# Eye Aspect Ratio (EAR) threshold below which the eye is considered closed
EAR_THRESHOLD = 0.20  # Adjust based on testing conditions

# Number of consecutive frames the EAR must be below the threshold to trigger the alarm
EAR_CONSECUTIVE_FRAMES = 38  # ~2 seconds if running at 24 FPS

# --- Yawn detection configuration ---
# Minimum vertical mouth opening (in pixels) to be considered a yawn
YAWN_THRESHOLD = 20

# Number of consecutive frames mouth must stay open to count as a yawn
YAWN_CONSECUTIVE_FRAMES = 25

# --- Head-nod detection configuration ---
# Minimum downward movement of the nose tip (in pixels) between consecutive
# frames that constitutes a nod.
NOD_THRESHOLD = 8

# Path to the alarm audio file (WAV/MP3)
ALARM_SOUND_PATH = "alarm.mp3"

# MediaPipe Face Mesh landmarks for the left and right eyes
# These six points are sufficient for EAR calculation (see MediaPipe documentation)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ============================================
# Helper Functions
# ============================================

def play_alarm_sound():
    """Play the alarm sound in a background thread.

    This function is intended to be run in a daemon thread so that it does not
    block the main loop. If the audio file is missing or cannot be played, the
    exception is caught and ignored so the main application continues running.
    """
    try:
        playsound(ALARM_SOUND_PATH)
    except Exception as e:
        # Log error (could be expanded to use logging module)
        print(f"[WARN] Unable to play alarm sound: {e}")


def calculate_ear(landmarks: np.ndarray, eye_indices: list) -> float:
    """Compute the Eye Aspect Ratio (EAR) for a given eye.

    Args:
        landmarks (np.ndarray): Array of shape (468, 2) containing the (x, y)
            pixel coordinates of all face landmarks.
        eye_indices (list): List of six landmark indices defining the eye.

    Returns:
        float: The computed EAR value for the specified eye.
    """
    # Extract the six (x, y) coordinates for the eye
    p1, p2, p3, p4, p5, p6 = landmarks[eye_indices]

    # Compute the two vertical distances
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    # Compute the horizontal distance
    horizontal = np.linalg.norm(p1 - p4)

    # Avoid division by zero
    if horizontal == 0:
        return 0.0

    # Eye Aspect Ratio formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)

# ============================================
# Main Drowsiness Detection Pipeline
# ============================================

def main():
    print("[INFO] Starting Driver Drowsiness Detection …")

    # Initialize MediaPipe Face Mesh with GPU delegate if available.
    # MediaPipe Python wheels include GPU support when available on the system.
    mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]

    # Aliases to keep type checkers happy for MediaPipe helpers
    mp_drawing_utils = mp.solutions.drawing_utils  # type: ignore[attr-defined]
    mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore[attr-defined]

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize the webcam stream (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # Frame counter for EAR below threshold
    counter = 0
    alarm_on = False

    # Initialize yawn counter
    yawn_counter = 0

    # Initialize previous nose y position
    prev_nose_y = 0

    # Main processing loop
    while True:
        success, frame = cap.read()
        if not success:
            print("[WARN] Frame capture failed, skipping …")
            continue

        frame_height, frame_width = frame.shape[:2]

        # Flip the frame horizontally for natural (mirror) viewing
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Improve performance by marking the frame as non-writeable to pass by reference
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True  # restore

        if results.multi_face_landmarks:
            # Assume the first face is the driver
            face_landmarks = results.multi_face_landmarks[0]

            # Convert landmark coordinates to pixel positions for EAR calculation
            landmarks = np.array(
                [(lm.x * frame_width, lm.y * frame_height) for lm in face_landmarks.landmark]
            )

            left_ear = calculate_ear(landmarks, LEFT_EYE_IDX)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_IDX)
            avg_ear = (left_ear + right_ear) / 2.0

            # -------------- Yawn Detection --------------
            upper_lip_y = landmarks[13][1]
            lower_lip_y = landmarks[14][1]
            mouth_open_dist = abs(lower_lip_y - upper_lip_y)

            # Update yawn counter
            if mouth_open_dist > YAWN_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

            yawn_detected = yawn_counter >= YAWN_CONSECUTIVE_FRAMES

            # -------------- Head Nod Detection --------------
            current_nose_y = landmarks[1][1]
            delta_y = 0 if prev_nose_y == 0 else (current_nose_y - prev_nose_y)
            nod_detected = delta_y > NOD_THRESHOLD
            prev_nose_y = current_nose_y

            # Draw facial landmarks for visualization (optional)
            mp_drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,  # type: ignore[attr-defined]
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),  # type: ignore[attr-defined]
            )

            # -------------- Integrate Detections --------------
            eye_closed = avg_ear < EAR_THRESHOLD

            if eye_closed:
                counter += 1
            else:
                counter = 0

            eye_alert = counter >= EAR_CONSECUTIVE_FRAMES

            # Determine if any alert condition met
            alert_condition = eye_alert or yawn_detected or nod_detected

            # Decide status text and color for UI feedback
            if alert_condition:
                if eye_alert:
                    status_text = "EYES CLOSED"
                elif yawn_detected:
                    status_text = "YAWN DETECTED"
                else:
                    status_text = "HEAD NOD DETECTED"
                color = (0, 0, 255)  # Red
            else:
                # Provide immediate feedback if eyes are briefly closed (blink)
                if eye_closed:
                    status_text = "Blinking"
                    color = (0, 255, 255)  # Yellow
                else:
                    status_text = "Alert"
                    color = (0, 255, 0)  # Green

            # Trigger alarm if needed
            if alert_condition and not alarm_on:
                alarm_on = True
                print(f"[ALERT] {status_text}! Triggering alarm …")
                threading.Thread(target=play_alarm_sound, daemon=True).start()
            elif not alert_condition:
                alarm_on = False

            # Display metrics and status
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.3f}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Mouth: {mouth_open_dist:.1f}",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                status_text,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        else:
            # Reset if no face is detected
            counter = 0
            alarm_on = False
            yawn_counter = 0
            prev_nose_y = 0
            cv2.putText(
                frame,
                "No face detected",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Show the frame
        cv2.imshow("Driver Drowsiness Detection", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    print("[INFO] Exiting application.")


if __name__ == "__main__":
    main()
