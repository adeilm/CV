"""Service entry point.

Run:
    python cv_pipeline.py        # uses CV_PORT from .env (default 8005)

Or with explicit uvicorn (e.g. for hot-reload during dev):
    uvicorn app.routes:app --host 0.0.0.0 --port 8005 --reload

NOTE: keep `--workers 1`. MediaPipe FaceMesh objects in app/signals.py
are module-level globals and are not thread-safe across processes.
"""
from app.routes import app
from app.config import PORT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
