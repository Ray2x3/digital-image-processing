"""Core image operations used by both the web UI and API."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

@dataclass
class ImageInfo:
    width: int
    height: int
    channels: int
    mode: str
    format: str
    file_size: int

def decode_image(data: bytes) -> tuple[np.ndarray, ImageInfo]:
    """Decode PNG/JPEG/BMP bytes as RGB and retain source metadata."""
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("無法讀取影像，請使用 PNG、JPG/JPEG 或 BMP 檔案。") from exc
    fmt = (image.format or "").upper()
    if fmt not in {"PNG", "JPEG", "BMP"}:
        raise ValueError("目前僅支援 PNG、JPG/JPEG 與 BMP 格式。")
    mode = image.mode
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb, ImageInfo(rgb.shape[1], rgb.shape[0], 3, mode, fmt, len(data))

def view_channels(rgb: np.ndarray, mode: str) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    if mode == "Original":
        return rgb, [(name, rgb[:, :, i]) for i, name in enumerate(("R", "G", "B"))]
    if mode in {"R", "G", "B"}:
        plane = rgb[:, :, {"R": 0, "G": 1, "B": 2}[mode]]
        return plane, [(mode, plane)]
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    idx = {"Y": 0, "U": 1, "V": 2}.get(mode)
    if idx is None:
        raise ValueError(f"未知顯示模式：{mode}")
    plane = yuv[:, :, idx]
    return plane, [(mode, plane)]

def apply_kernel(rgb: np.ndarray, rows: list[list[float]], normalize: bool = False) -> np.ndarray:
    """Apply a square odd-sized kernel with replicated borders and uint8 clipping."""
    kernel = np.asarray(rows, dtype=np.float32)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
        raise ValueError("卷積核必須是奇數邊長的方陣（例如 3×3 或 5×5）。")
    if kernel.shape[0] > 15:
        raise ValueError("卷積核最大支援 15×15。")
    if not np.isfinite(kernel).all():
        raise ValueError("卷積核只能包含有限數值。")
    if normalize:
        total = float(kernel.sum())
        if abs(total) < 1e-12:
            raise ValueError("卷積核總和為 0，無法進行正規化。")
        kernel /= total
    # OpenCV filter2D computes correlation; flipping both axes gives convolution.
    kernel = np.flip(kernel, axis=(0, 1)).copy()
    return cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REPLICATE)

def psnr(original: np.ndarray, processed: np.ndarray) -> float:
    mse = float(np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * float(np.log10(255.0 ** 2 / mse))

def detect_faces(rgb: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Detect with OpenCV Haar Cascade; this detector does not expose confidence."""
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("無法載入 OpenCV Haar 人臉模型。")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    result, faces = rgb.copy(), []
    for x, y, w, h in boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        faces.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
    return result, faces
