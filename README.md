[![Support Palestine](https://raw.githubusercontent.com/Ademking/Support-Palestine/main/Support-Palestine.svg)](https://www.map.org.uk)

# Driver Monitoring System (DMS) 🚗💤
![Face-Mask-Detection](https://miro.medium.com/v2/resize:fit:1400/1*gV-Wcn3-bAx5DBELHEPtdg.jpeg)

This project implements a real-time **Driver Drowsiness Detection System** using `MediaPipe` for facial landmark tracking and `OpenCV` for video processing. It helps detect signs of driver fatigue such as **eye closure**, **yawning**, and **head nodding**, and triggers an audible alarm if any drowsiness condition is detected.

---

## 🔧 Features

- 👁️ **Eye Aspect Ratio (EAR)** based detection for prolonged eye closure.
- 😮 **Yawn detection** using mouth opening distance.
- 🤕 **Head nod detection** based on nose movement.
- 🔔 Plays an alarm when a drowsiness condition is met.
- 🖼️ Real-time video feed with on-screen alerts and metrics.

---

## 📁 Project Structure

```
.
├── pipeline.py           # Main Python script for detection
├── alarm.mp3             # Alarm sound file (you must provide this)
└── README.md             # This file
```

---

## ▶️ How It Works

The script performs the following:

1. Captures webcam video using `cv2.VideoCapture`.
2. Uses **MediaPipe FaceMesh** to extract facial landmarks.
3. Computes:
   - **EAR (Eye Aspect Ratio)** for eye closure
   - **Lip distance** for yawning
   - **Nose Y movement** for head nodding
4. If a drowsiness event is detected:
   - Shows alert message on screen
   - Plays an alarm sound using `playsound`

---

## 🧪 Drowsiness Metrics

| Condition       | Trigger                                 |
|----------------|------------------------------------------|
| Eyes Closed     | EAR < `0.20` for 38 consecutive frames  |
| Yawn Detected   | Mouth open > `20px` for 25 frames       |
| Head Nod        | Nose tip drops > `8px` suddenly         |

---

## 🚀 How to Run

1. **Install dependencies:**

```bash
pip install opencv-python mediapipe playsound numpy
```

2. **Place an alarm sound file** in the root directory:

```bash
alarm.mp3
```

3. **Run the script:**

```bash
python pipeline.py
```

4. **Press `q`** to quit the window.

---

## 📸 Visual Output

- Displays EAR and mouth distance on screen
- Highlights detection status: `EYES CLOSED`, `YAWN DETECTED`, `HEAD NOD DETECTED`, or `Alert`
- Shows "No face detected" if no driver is in view

---

## 📌 Requirements

- Python 3.7+
- Webcam
- `alarm.mp3` sound file

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Credits

Developed using:
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [playsound](https://github.com/TaylorSMarks/playsound)
