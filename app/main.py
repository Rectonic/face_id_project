import argparse
import base64
import math
import pathlib
import shutil
import ssl
import threading
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import certifi
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
PROTOTXT_PATH = MODEL_DIR / "deploy.prototxt"
MODEL_PATH = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
ARCFACE_MODEL_PATH = MODEL_DIR / "arcface_w600k_r50.onnx"
ARCFACE_ARCHIVE_PATH = MODEL_DIR / "buffalo_l.zip"
ARCFACE_MODEL_MEMBER = "w600k_r50.onnx"
KNOWN_FACES_DIR = BASE_DIR / "templates" / "faces"
VALID_FACE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# URLs hosted by the OpenCV team for the pretrained face detector and ArcFace archive
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)
ARCFACE_ARCHIVE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

EYE_CASCADE_PATH = pathlib.Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
EYE_CASCADE = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))
if EYE_CASCADE.empty():  # pragma: no cover - fallback if cascade is unavailable
    EYE_CASCADE = None


face_net_lock = threading.Lock()
arcface_lock = threading.Lock()
known_faces_lock = threading.Lock()
known_faces_snapshot: Tuple[Tuple[str, float, int], ...] = ()
known_faces: Dict[str, Dict[str, object]] = {}


def _download_file(url: str, target: pathlib.Path) -> None:
    """Download helper that pins CA bundle for reliable SSL verification."""
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context) as response:
        target.write_bytes(response.read())


def ensure_model_files() -> None:
    """Download the face detection and recognition model files if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not PROTOTXT_PATH.exists():
        _download_file(PROTOTXT_URL, PROTOTXT_PATH)
    if not MODEL_PATH.exists():
        _download_file(MODEL_URL, MODEL_PATH)
    need_embedding = False
    if not ARCFACE_MODEL_PATH.exists():
        need_embedding = True
    else:
        try:
            if ARCFACE_MODEL_PATH.stat().st_size < 1024 * 1024:
                need_embedding = True
        except OSError:
            need_embedding = True
    if need_embedding:
        _download_file(ARCFACE_ARCHIVE_URL, ARCFACE_ARCHIVE_PATH)
        with zipfile.ZipFile(ARCFACE_ARCHIVE_PATH, "r") as archive:
            if ARCFACE_MODEL_MEMBER not in archive.namelist():
                raise FileNotFoundError(
                    f"{ARCFACE_MODEL_MEMBER} not found in ArcFace archive."
                )
            with archive.open(ARCFACE_MODEL_MEMBER) as source, ARCFACE_MODEL_PATH.open(
                "wb"
            ) as target:
                shutil.copyfileobj(source, target)
        try:
            ARCFACE_ARCHIVE_PATH.unlink()
        except OSError:
            pass


def load_model() -> cv2.dnn_Net:
    """Load the OpenCV DNN face detector."""
    ensure_model_files()
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(MODEL_PATH))
    return net


def load_embedding_model() -> ort.InferenceSession:
    """Load the ArcFace embedding model with ONNX Runtime."""
    ensure_model_files()
    session = ort.InferenceSession(
        str(ARCFACE_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    return session


def detect_faces(
    image: np.ndarray, net: cv2.dnn_Net, confidence_threshold: float = 0.5
) -> List[Tuple[int, int, int, int, float]]:
    """
    Run the face detector on a BGR image.

    Returns a list of (x, y, w, h, confidence) tuples using pixel coordinates.
    """
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    with face_net_lock:
        net.setInput(blob)
        detections = net.forward()

    results: List[Tuple[int, int, int, int, float]] = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")

        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w - 1, end_x)
        end_y = min(h - 1, end_y)

        width = end_x - start_x
        height = end_y - start_y
        if width <= 0 or height <= 0:
            continue

        results.append((start_x, start_y, width, height, float(confidence)))

    return results


def select_primary_detection(
    detections: List[Tuple[int, int, int, int, float]]
) -> Optional[Tuple[int, int, int, int, float]]:
    if not detections:
        return None
    return max(detections, key=lambda det: det[2] * det[3])


def align_face(face_image: np.ndarray) -> np.ndarray:
    if face_image.size == 0 or EYE_CASCADE is None:
        return face_image
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    min_eye_size = (
        max(12, face_image.shape[1] // 8),
        max(12, face_image.shape[0] // 8),
    )
    eyes = EYE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=min_eye_size,
    )
    if len(eyes) < 2:
        return face_image
    # Choose the two eyes with the widest separation (likely left/right).
    eyes = sorted(eyes, key=lambda e: e[0])
    left_eye = eyes[0]
    right_eye = eyes[-1]
    left_center = (
        left_eye[0] + left_eye[2] / 2.0,
        left_eye[1] + left_eye[3] / 2.0,
    )
    right_center = (
        right_eye[0] + right_eye[2] / 2.0,
        right_eye[1] + right_eye[3] / 2.0,
    )
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    if abs(dx) < 1e-3:
        return face_image
    angle = math.degrees(math.atan2(dy, dx))
    center = (
        (left_center[0] + right_center[0]) / 2.0,
        (left_center[1] + right_center[1]) / 2.0,
    )
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(
        face_image,
        rotation_matrix,
        (face_image.shape[1], face_image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned


def normalize_lighting(face_image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
    y_channel = cv2.equalizeHist(ycrcb[:, :, 0])
    ycrcb[:, :, 0] = y_channel
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def crop_face(image: np.ndarray, detection: Tuple[int, int, int, int, float]) -> Optional[np.ndarray]:
    x, y, w, h, _ = detection
    padding_x = int(w * 0.2)
    padding_y = int(h * 0.2)
    x0 = max(0, x - padding_x)
    y0 = max(0, y - padding_y)
    x1 = min(image.shape[1], x + w + padding_x)
    y1 = min(image.shape[0], y + h + padding_y)
    face_image = image[y0:y1, x0:x1]
    if face_image.size == 0:
        return None
    aligned = align_face(face_image)
    enhanced = normalize_lighting(aligned)
    return enhanced


def prepare_arcface_input(face_image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_image, (112, 112))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 127.5) - 1.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def compute_embedding_from_face(face_image: np.ndarray) -> Optional[np.ndarray]:
    if face_image is None or face_image.size == 0:
        return None
    input_blob = prepare_arcface_input(face_image)
    with arcface_lock:
        embedding = arcface_session.run(
            None, {arcface_input_name: input_blob}
        )[0]
    embedding = embedding.reshape(-1)
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        return None
    return embedding / norm


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def encode_image_to_data_url(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"


def annotate_image(
    image: np.ndarray,
    detection: Tuple[int, int, int, int, float],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    secondary_text: Optional[str] = None,
) -> np.ndarray:
    annotated = image.copy()
    x, y, w, h, confidence = detection
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    if secondary_text is None:
        text = f"{label} ({confidence*100:.1f}%)"
    else:
        text = f"{label} {secondary_text}"
    cv2.putText(
        annotated,
        text,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    return annotated


def build_face_embedding(
    image: np.ndarray, detector_threshold: float = 0.4
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int, float]]]:
    detections = detect_faces(image, face_net, detector_threshold)
    primary = select_primary_detection(detections)
    if primary is None:
        return None, None
    face_image = crop_face(image, primary)
    embedding = compute_embedding_from_face(face_image)
    if embedding is None:
        return None, None
    return embedding, primary


def format_display_name(value) -> str:
    if isinstance(value, pathlib.Path):
        raw = value.stem
    else:
        raw = str(value)
    raw = raw.replace("_", " ").replace("-", " ").strip()
    if not raw:
        return str(value)
    return " ".join(part.capitalize() for part in raw.split())


def scan_faces_inventory() -> Tuple[Tuple[str, float, int], ...]:
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    inventory: List[Tuple[str, float, int]] = []
    for image_path in KNOWN_FACES_DIR.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_FACE_EXTENSIONS:
            continue
        try:
            stats = image_path.stat()
        except OSError:
            continue
        relative = str(image_path.relative_to(KNOWN_FACES_DIR))
        inventory.append((relative, stats.st_mtime, stats.st_size))
    return tuple(sorted(inventory))


def load_known_faces() -> Dict[str, Dict[str, object]]:
    grouped_embeddings: Dict[str, Dict[str, object]] = {}
    for image_name, _, _ in scan_faces_inventory():
        image_path = KNOWN_FACES_DIR / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        embedding, _ = build_face_embedding(image, detector_threshold=0.4)
        if embedding is None:
            continue
        relative_path = pathlib.Path(image_name)
        if relative_path.parent == pathlib.Path("."):
            person_key = relative_path.stem
        else:
            person_key = relative_path.parts[0]
        entry = grouped_embeddings.setdefault(
            person_key,
            {
                "display_name": format_display_name(person_key),
                "embeddings": [],
                "sources": [],
            },
        )
        entry["embeddings"].append(embedding)
        entry["sources"].append(image_name)

    aggregated: Dict[str, Dict[str, object]] = {}
    for key, info in grouped_embeddings.items():
        embeddings = np.vstack(info["embeddings"])
        prototype = embeddings.mean(axis=0)
        norm = np.linalg.norm(prototype)
        if not np.isfinite(norm) or norm == 0:
            continue
        prototype /= norm
        aggregated[key] = {
            "embedding": prototype,
            "display_name": info["display_name"],
            "filenames": info["sources"],
            "preview": info["sources"][0],
            "count": len(info["sources"]),
        }
    return aggregated


def reload_known_faces() -> None:
    global known_faces_snapshot
    faces = load_known_faces()
    inventory = scan_faces_inventory()
    with known_faces_lock:
        known_faces.clear()
        known_faces.update(faces)
        known_faces_snapshot = inventory


def ensure_known_faces_loaded() -> None:
    global known_faces_snapshot
    current_inventory = scan_faces_inventory()
    with known_faces_lock:
        if current_inventory == known_faces_snapshot:
            return
    faces = load_known_faces()
    with known_faces_lock:
        known_faces.clear()
        known_faces.update(faces)
        known_faces_snapshot = current_inventory


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
face_net = load_model()
arcface_session = load_embedding_model()
arcface_input_name = arcface_session.get_inputs()[0].name
DEFAULT_MATCH_THRESHOLD = 0.45
reload_known_faces()


@app.route("/faces/<path:filename>")
def serve_face_image(filename: str):
    ensure_known_faces_loaded()
    safe_path = pathlib.Path(filename)
    if ".." in safe_path.parts:
        abort(404)
    target = (KNOWN_FACES_DIR / safe_path).resolve()
    if not str(target).startswith(str(KNOWN_FACES_DIR.resolve())) or not target.exists():
        abort(404)
    return send_from_directory(str(KNOWN_FACES_DIR), filename)


@app.route("/")
def index():
    ensure_known_faces_loaded()
    with known_faces_lock:
        faces_for_template: List[Dict[str, object]] = []
        for info in sorted(
            known_faces.values(), key=lambda item: item["display_name"].lower()
        ):
            preview = info.get("preview")
            faces_for_template.append(
                {
                    "name": info["display_name"],
                    "count": info.get("count", 1),
                    "url": url_for("serve_face_image", filename=preview)
                    if preview
                    else None,
                }
            )
    return render_template(
        "index.html",
        known_faces=faces_for_template,
        default_match_threshold=DEFAULT_MATCH_THRESHOLD,
    )


@app.post("/api/detect")
def api_detect():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded under field 'image'."}), 400

    file_storage = request.files["image"]
    file_bytes = file_storage.read()

    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    threshold_value = request.form.get("threshold", type=float)
    if threshold_value is None:
        confidence_threshold = 0.5
    else:
        confidence_threshold = float(np.clip(threshold_value, 0.1, 0.99))

    np_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Unable to decode the uploaded image."}), 400

    detections = detect_faces(image, face_net, confidence_threshold)

    response_boxes = []
    (img_h, img_w) = image.shape[:2]
    for (x, y, width, height, confidence) in detections:
        response_boxes.append(
            {
                "x": x / img_w,
                "y": y / img_h,
                "width": width / img_w,
                "height": height / img_h,
                "confidence": confidence,
            }
        )
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        text = f"{confidence*100:.1f}%"
        cv2.putText(
            image,
            text,
            (x, y - 10 if y - 10 > 10 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{base64_image}"

    return jsonify({"boxes": response_boxes, "image": data_url})


@app.post("/api/match")
def api_match_face():
    ensure_known_faces_loaded()
    with known_faces_lock:
        known_items = list(known_faces.items())

    if not known_items:
        return jsonify(
            {
                "error": (
                    "No known faces loaded. Add portrait images under "
                    "templates/faces and reload the page."
                )
            }
        ), 400

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded under field 'image'."}), 400

    file_bytes = request.files["image"].read()
    if not file_bytes:
        return jsonify({"error": "Uploaded frame is empty."}), 400

    match_threshold = request.form.get("matchThreshold", type=float)
    if match_threshold is None:
        threshold_value = DEFAULT_MATCH_THRESHOLD
    else:
        threshold_value = float(np.clip(match_threshold, 0.1, 0.99))

    np_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Unable to decode the uploaded frame."}), 400

    candidate_embedding, detection = build_face_embedding(image, detector_threshold=0.4)
    if candidate_embedding is None or detection is None:
        return jsonify({"error": "No face detected in the frame."}), 422

    best_similarity = -1.0
    best_info: Optional[Dict[str, object]] = None
    for _, info in known_items:
        similarity = cosine_similarity(info["embedding"], candidate_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_info = info

    if best_info is None:
        return jsonify({"error": "No embeddings are available to compare."}), 500

    is_match = best_similarity >= threshold_value
    match_name = best_info["display_name"] if is_match else None
    label = best_info["display_name"] if is_match else "UNKNOWN"
    color = (0, 255, 0) if is_match else (0, 0, 255)
    secondary_text = f"(sim {best_similarity*100:.1f}%)"
    annotated = annotate_image(image, detection, label, color, secondary_text=secondary_text)
    greeting = (
        f"Hello, {best_info['display_name']}!"
        if is_match
        else "Face not recognized. Try again."
    )
    sample_count = int(best_info.get("count", 1))
    samples = best_info.get("filenames", [])

    img_h, img_w = image.shape[:2]
    normalized_box = {
        "x": detection[0] / img_w,
        "y": detection[1] / img_h,
        "width": detection[2] / img_w,
        "height": detection[3] / img_h,
        "confidence": detection[4],
    }

    return jsonify(
        {
            "match": is_match,
            "name": match_name,
            "similarity": best_similarity,
            "threshold": threshold_value,
            "box": normalized_box,
            "image": encode_image_to_data_url(annotated),
            "greeting": greeting,
            "closest": {
                "name": best_info["display_name"],
                "similarity": best_similarity,
            },
            "sample_count": sample_count,
            "samples": samples,
        }
    )


def create_app() -> Flask:
    """Factory for WSGI servers."""
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    app.run(debug=True, host=args.host, port=args.port)
