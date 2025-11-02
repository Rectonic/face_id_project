# Face ID Playground

A tiny web application that wraps OpenCV's face detector and an ArcFace embedding model so you can prototype Face ID-style flows entirely on your machine. Drop a collection of portraits into `templates/faces/`, start your webcam, and watch the app greet anyone it recognizes. You can still experiment with plain detection to inspect bounding boxes and confidence values.

## Features

- Uses OpenCV's high-quality DNN face detector (`res10_300x300_ssd_iter_140000.caffemodel`).
- Extracts 512-D ArcFace embeddings with ONNX Runtime for accurate identity separation.
- Auto-builds an in-memory gallery from portrait images stored in `templates/faces/`, averaging all photos per person for stability.
- Live webcam panel continuously matches frames while the camera is active and overlays the best match with a friendly greeting.
- Adjustable similarity and detection thresholds so you can tune sensitivity quickly.
- REST API exposes `/api/detect`, `/api/match`, and `/faces/<filename>` for integration tests or headless workflows.

## Getting Started

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:

   ```bash
   python app/main.py
   ```

   The server starts on `http://localhost:8000`.

4. Open your browser to `http://localhost:8000`. The first visit downloads the detector and embedding weights under `models/`.

5. Drop portrait images (JPG/PNG/WebP/BMP) inside `templates/faces/`. You can:
   - Place a single photo directly under `templates/faces/` (its file name supplies the display label).
   - Create subfolders such as `templates/faces/alice/` and add multiple photos per person—the app averages their embeddings automatically.
   Refresh the page to see the gallery update.

## Face Matching Workflow

1. **Curate the gallery** – place headshots in `templates/faces/`. The server automatically recomputes embeddings (and averages all photos per identity) whenever the directory changes.
2. **Start the webcam** – click **Start Camera** in the live match panel. Allow browser camera access when prompted.
3. **Let the live matcher run** – the app continuously captures frames, compares them to every embedding, and updates the overlay with similarity metrics and a greeting. Use the **Check Now** button if you want to trigger an immediate re-check.
4. **Experiment freely** – use the detection sandbox to verify bounding boxes and scores on arbitrary images.

## REST Endpoints

- `POST /api/match` – multipart form with `image` and optional `matchThreshold` (0–1). Responds with `match`, `name`, raw cosine similarity, and an annotated frame.
- `POST /api/detect` – multipart form with `image` and optional `threshold`. Returns normalized boxes and an annotated preview.
- `GET /faces/<filename>` – serves gallery images stored under `templates/faces/`.

## Notes

- Model weights download from the official OpenCV and InsightFace mirrors. Keep an internet connection the first time you run the app so the files cache locally (the ArcFace model is ~170&nbsp;MB).
- The similarity comparison defaults to 0.45 (a balanced ArcFace threshold). Increase it for stricter matches or lower it for leniency.
- Embeddings are recomputed from disk whenever the gallery directory changes, so the app stays in sync with new or updated photos.
- The UI uses plain browser APIs. For production, consider HTTPS to avoid camera permission friction on non-localhost hosts.
- `/api/match` requests run through a small worker queue so the ArcFace model processes one frame at a time; clients automatically back off when they receive a 429 "busy" response.

## Next Steps

- Swap in a different detector (e.g., RetinaFace or YOLO-based models) by adjusting the weights and inference pipeline in `app/main.py`.
- Persist embeddings alongside user identities in a database so the match endpoint can return structured records or audit trails.
- Add liveness detection or temporal smoothing if you plan to exercise stronger spoofing resistance.
