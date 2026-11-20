import os
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ml_engine import FYPPulseEngine

app = FastAPI(title="FYP-Pulse: Early Warning System")
engine = FYPPulseEngine()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.api_route("/favicon.ico", methods=["GET", "HEAD"])
def favicon():
    return FileResponse(os.path.join(STATIC_DIR, "favicon.svg"), media_type="image/svg+xml")


class PredictRequest(BaseModel):
    data: Dict[str, Any]


class WhatIfRequest(BaseModel):
    baseline: Dict[str, Any]
    levers: Dict[str, Any]


@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/overview")
def get_overview():
    if not engine.ready:
        return {"ready": False, "message": "Upload a CSV file or place one in the project folder."}
    return engine.get_overview_payload()


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid .csv file.")
    content = await file.read()
    try:
        return engine.fit_from_csv_bytes(content, source_name=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {exc}")


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not engine.ready:
        raise HTTPException(status_code=400, detail="No dataset loaded yet.")
    return engine.predict_group(req.data)


@app.get("/api/watchlist")
def watchlist(k: int = Query(30, ge=10, le=100)):
    if not engine.ready:
        raise HTTPException(status_code=400, detail="No dataset loaded yet.")
    return engine.get_watchlist(k=k)


@app.get("/api/watchlist/download")
def download_watchlist(k: int = Query(30, ge=10, le=100)):
    if not engine.ready:
        raise HTTPException(status_code=400, detail="No dataset loaded yet.")
    csv_text = engine.get_watchlist_csv(k=k)
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=fyp_pulse_watchlist_top_{k}.csv"},
    )


@app.post("/api/whatif")
def whatif(req: WhatIfRequest):
    if not engine.ready:
        raise HTTPException(status_code=400, detail="No dataset loaded yet.")
    return engine.simulate_whatif(req.baseline, req.levers)


@app.post("/api/whatif/best-lever")
def best_lever(req: PredictRequest):
    if not engine.ready:
        raise HTTPException(status_code=400, detail="No dataset loaded yet.")
    return engine.find_best_lever(req.data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)