# Face ID Playground

Face ID Playground is a self-contained Flask app for experimenting with live face detection and recognition. It bundles OpenCV’s SSD face detector with an InsightFace ArcFace embedding model so you can prototype “Face ID” style flows locally. Drop portraits into `templates/faces/`, run the server, and the browser UI lets you try detection, similarity matching, and live webcam checks.

---

## Tech Stack

- **Backend**: Flask 3 serving HTML templates and JSON APIs.
- **Computer Vision**: Customly trained and built version of OpenCV DNN face detector + ArcFace ONNX model executed with ONNX Runtime.
- **Data**: Face gallery sourced from images on disk (`templates/faces/`), embeddings cached in memory.
- **Frontend**: Vanilla HTML/CSS/JS under `templates/` and `static/`.

---

## Prerequisites

- Python 3.10+ with `pip`.
- Internet access on first run so the model weights can download (~180 MB total).
- A webcam (optional) for the live match panel.

If you are working on macOS/Linux, consider installing `python3-venv` or `virtualenv` for isolated environments.

---

## Quick Start

```bash
# 1. (Optional) create an isolated environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install third-party packages
pip install -r requirements.txt

# 3. Launch the development server
python app/main.py --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` in a browser. The first request downloads the detector and ArcFace weights into `models/`.

---

## Project Layout

```
app/
  main.py              # Flask app, background workers, detection/matching logic
models/                # Auto-populated with detector and ArcFace weights
static/                # CSS/JS/assets for the browser UI
templates/
  faces/               # Gallery of known faces (your photos go here)
  index.html           # Main web UI
requirements.txt       # Python dependencies
```

---

## Customizing the Experience

- **Gallery management**: Upload JPG/PNG/WebP/BMP images from the homepage, capture snapshots directly from the webcam, or drop them into `templates/faces/`.  
  - The upload form stores portraits under `templates/faces/<slug>/` and refreshes the gallery instantly.  
  - Manual option: place files yourself (`templates/faces/alice.jpg` or `templates/faces/alice/portrait1.jpg`, etc.). The app averages embeddings per person.  
  - Delete entries straight from the gallery when you need to prune the dataset.
- **Webcam snapshots**: В интерфейсе нажмите «Сделать снимок», исправьте имя в всплывающем окне и сохраните — кадр попадет в галерею сразу после обработки.
- **Thresholds**: Default detection threshold is 0.5; default match threshold is 0.45. Both can be adjusted from the UI for quick experiments. To hard-code new defaults, edit the constants near the top of `app/main.py`.
- **Model swaps**: Update `PROTOTXT_URL`, `MODEL_URL`, or `ARCFACE_ARCHIVE_URL` inside `app/main.py` if you want to point to different pretrained models. Adjust preprocessing/postprocessing as required by the new models.
- **Serving options**: For production, wrap the Flask app with a WSGI server such as `gunicorn` and run behind HTTPS so browsers allow camera access without complaints.


---

## API Endpoints

- `POST /api/detect`  
  Send multipart form data with `image=<file>` and optional `threshold=<float 0-1>`. Response includes normalized bounding boxes and an annotated preview image (Base64).
- `POST /api/match`  
  Send multipart form data with `image=<file>` and optional `matchThreshold=<float 0-1>`. Response returns the best match, cosine similarity, and the annotated preview. A lightweight queue serializes match jobs to keep ONNX Runtime responsive.
- `POST /api/upload-face`  
  Send multipart form data with `person=<name>` and one or more `images=<file>` entries (snapshots captured in the browser are accepted via `capture`). The server enqueues an embedding reload and responds immediately with `202 Accepted`; call `GET /api/known-faces` after a short delay to see the updated gallery.
- `GET /api/known-faces`  
  Returns the current gallery metadata (display name, photo count, preview URL) so clients can refresh without a full reload.
- `DELETE /api/face/<slug>`  
  Removes all images for the given person slug, triggers an asynchronous reload, and responds with `202 Accepted`.
- `GET /faces/<filename>`  
  Serves gallery assets for embedding in the UI or for verification.

---

## Deployment

### Docker

```bash
docker build -t face-id-playground .
docker run --rm -p 8000:8000 -v $(pwd)/templates/faces:/app/templates/faces face-id-playground
```

Mount `templates/faces` (or another writable directory) so uploaded portraits persist across restarts. If you want GPU acceleration, swap the base image in `Dockerfile` for an ONNX Runtime GPU build and update the provider list in `app/main.py`.

### Bare Metal / VM

Run the Flask app behind a process manager (Gunicorn, systemd, supervisord) and terminate TLS with a reverse proxy (Nginx, Caddy, Traefik). Keep HTTPS enabled so browsers grant camera access without warnings. Ensure the process user can write to `templates/faces/` and `models/`.

---

## Development Tips

- **Hot reload**: Flask runs in debug mode by default (`app.run(debug=True, ...)`) so code changes reload automatically.
- **Updating dependencies**: Add new Python packages to `requirements.txt` and reinstall (`pip install -r requirements.txt`).
- **Testing new photos**: After adding or editing files in `templates/faces/`, refresh the web page. The server re-indexes and recomputes embeddings automatically.
- **Cleaning downloads**: Delete files under `models/` if you need to force a fresh model download.

---

## Troubleshooting

- **Model download failures**: Ensure outbound HTTPS access. If needed, download the files manually and place them in `models/` using the filenames referenced in `app/main.py`.
- **Camera blocked**: Browsers only grant webcam access over HTTPS or `localhost`. Use `https://localhost:<port>` with a dev certificate or keep testing on the same machine as the server.
- **High CPU usage**: Matching work is serialized through a queue. If you expect heavy load, consider increasing the worker count and splitting requests across processes.

Enjoy exploring face detection and recognition locally!
