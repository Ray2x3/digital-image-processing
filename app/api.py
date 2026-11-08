"""FastAPI service exposing image metadata, filtering and face detection."""
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from .imaging import apply_kernel, decode_image, detect_faces, psnr

app = FastAPI(title="Image Lab API", version="0.1.0")

class FilterRequest(BaseModel):
    kernel: list[list[float]] = Field(min_length=1)
    normalize: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/image/info")
async def image_info(file: UploadFile = File(...)):
    try:
        _, info = decode_image(await file.read())
        return info
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/image/filter")
async def image_filter(kernel: FilterRequest, file: UploadFile = File(...)):
    try:
        rgb, _ = decode_image(await file.read())
        filtered = apply_kernel(rgb, kernel.kernel, kernel.normalize)
        return {"psnr_db": psnr(rgb, filtered), "width": int(rgb.shape[1]), "height": int(rgb.shape[0])}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/image/faces")
async def image_faces(file: UploadFile = File(...)):
    try:
        rgb, _ = decode_image(await file.read())
        _, faces = detect_faces(rgb)
        return {"count": len(faces), "faces": faces, "confidence": None,
                "note": "Haar Cascade does not provide calibrated confidence scores."}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
