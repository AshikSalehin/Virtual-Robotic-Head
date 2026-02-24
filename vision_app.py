import cv2
import uuid
import requests
import numpy as np
from fastapi import FastAPI
from ultralytics import YOLO
import mediapipe as mp
import face_recognition
from sklearn.cluster import KMeans
from collections import Counter

app = FastAPI()

# ---------------- LOAD MODELS ----------------
yolo = YOLO("yolov8n.pt")

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.6)

# ---------------- STORAGE ----------------
known_face_encodings = []
known_face_profiles = []  # {id, name, description}

unknown_faces = {}  # temp_id -> encoding


# ---------------- UTILITIES ----------------
def dominant_color(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=3, n_init=10).fit(img)
    counts = Counter(kmeans.labels_)
    color = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
    r, g, b = color.astype(int)

    if r > g and r > b:
        return "Red"
    if g > r and g > b:
        return "Green"
    if b > r and b > g:
        return "Blue"
    if r > 200 and g > 200 and b > 200:
        return "White"
    if r < 50 and g < 50 and b < 50:
        return "Black"
    return "Mixed"


def dress_type_from_crop(crop):
    h, w, _ = crop.shape
    if h > w:
        return "Dress"
    return "Shirt / T-Shirt"


# ---------------- CORE FRAME ANALYSIS ----------------
def process_frame(frame):
    detections = yolo(frame, conf=0.4, verbose=False)[0]

    for box, cls in zip(detections.boxes.xyxy, detections.boxes.cls):
        if yolo.names[int(cls)] != "person":
            continue

        x1, y1, x2, y2 = map(int, box)
        person = frame[y1:y2, x1:x2]

        # ---- Dress ----
        torso = person[int(0.25 * person.shape[0]):int(0.65 * person.shape[0]), :]
        dress_color = dominant_color(torso)
        dress_type = dress_type_from_crop(torso)
        dress = f"{dress_color} {dress_type}"

        # ---- Face Detection ----
        rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
        faces = face_detector.process(rgb)

        if not faces.detections:
            return

        bbox = faces.detections[0].location_data.relative_bounding_box
        h, w, _ = person.shape
        fx1 = int(bbox.xmin * w)
        fy1 = int(bbox.ymin * h)
        fx2 = fx1 + int(bbox.width * w)
        fy2 = fy1 + int(bbox.height * h)

        face_crop = person[fy1:fy2, fx1:fx2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(face_rgb)

        if not enc:
            return

        encoding = enc[0]

        # ---- Known Face? ----
        if known_face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, encoding, tolerance=0.5
            )

            if True in matches:
                idx = matches.index(True)
                profile = known_face_profiles[idx]

                payload = {
                    "ID": profile["id"],
                    "Dress": dress,
                    "Name": profile["name"],
                    "Description": profile["description"]
                }

                requests.post("http://localhost:8000/known_face", json=payload)
                return

        # ---- New Face ----
        temp_id = str(uuid.uuid4())[:8]
        unknown_faces[temp_id] = encoding

        payload = {
            "ID": temp_id,
            "Dress": dress
        }

        requests.post("http://localhost:8000/new_face", json=payload)


# ---------------- STREAM ENDPOINT ----------------
@app.post("/process_frame")
def receive_frame(frame: bytes):
    npimg = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    process_frame(img)
    return {"status": "processed"}


# ---------------- SAVE / UPDATE FACE ----------------
@app.post("/save_face")
def save_face(data: dict):
    temp_id = data["ID"]
    name = data["Name"]
    description = data["Description"]

    if temp_id not in unknown_faces:
        return {"error": "Invalid ID"}

    known_face_encodings.append(unknown_faces[temp_id])
    known_face_profiles.append({
        "id": temp_id,
        "name": name,
        "description": description
    })

    del unknown_faces[temp_id]
    return {"status": "face_saved"}
