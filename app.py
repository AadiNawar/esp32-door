from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import os
import pickle

app = Flask(__name__)

# =============================================
# Known faces are stored in /known_faces folder
# Each file should be named after the person
# e.g. aadi.jpg, john.jpg
# =============================================
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "encodings.pkl"

known_encodings = []
known_names     = []

def load_known_faces():
    """Load and encode all faces from known_faces directory."""
    global known_encodings, known_names

    # Use cached encodings if available (faster startup)
    if os.path.exists(ENCODINGS_FILE):
        print("Loading cached encodings...")
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            known_encodings = data["encodings"]
            known_names     = data["names"]
        print(f"Loaded {len(known_names)} known face(s): {known_names}")
        return

    # Otherwise encode from images
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print("No known_faces directory found — created empty one.")
        return

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]  # filename without extension = person's name
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"Loaded face: {name}")
            else:
                print(f"Warning: No face found in {filename}, skipping.")

    # Cache encodings for faster future startups
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print(f"Encoded and cached {len(known_names)} face(s).")


@app.route("/check", methods=["POST"])
def check_face():
    """
    POST a JPEG image to this endpoint.
    Returns JSON: { "decision": "OPEN", "name": "aadi" }
                  { "decision": "DENY", "name": "unknown" }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_array = np.frombuffer(file.read(), dtype=np.uint8)

    import cv2
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find faces in the snapshot
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    if not face_encodings:
        print("No face detected in snapshot.")
        return jsonify({"decision": "DENY", "name": "no_face"})

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        distances = face_recognition.face_distance(known_encodings, encoding)

        if True in matches:
            best_index = int(np.argmin(distances))
            name = known_names[best_index]
            confidence = round((1 - distances[best_index]) * 100, 1)
            print(f"Recognized: {name} ({confidence}% confidence) → OPEN")
            return jsonify({"decision": "OPEN", "name": name, "confidence": confidence})

    print("Face detected but not recognized → DENY")
    return jsonify({"decision": "DENY", "name": "unknown"})


@app.route("/reload", methods=["POST"])
def reload_faces():
    """Call this endpoint to reload known faces without restarting the server."""
    if os.path.exists(ENCODINGS_FILE):
        os.remove(ENCODINGS_FILE)
    load_known_faces()
    return jsonify({"status": "reloaded", "known": known_names})


@app.route("/status", methods=["GET"])
def status():
    """Health check — also shows who is authorized."""
    return jsonify({"status": "ok", "authorized": known_names})

# Load faces when app starts (works with gunicorn too)
with app.app_context():
    load_known_faces()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
