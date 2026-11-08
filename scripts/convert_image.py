"""Convert a JPEG to PNG and BMP, verifying decoded pixels before keeping outputs."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from PIL import Image


def convert_jpeg(input_path: Path, output_dir: Path | None = None) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到輸入檔案：{input_path}")

    with Image.open(input_path) as source:
        if source.format != "JPEG":
            raise ValueError(f"輸入檔不是 JPEG（偵測格式：{source.format}）。")
        source.load()
        source_rgb = source.convert("RGB")

    destination = (output_dir or input_path.parent).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    targets = [destination / f"{input_path.stem}.png", destination / f"{input_path.stem}.bmp"]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError("為避免覆寫既有檔案，已中止：" + ", ".join(map(str, existing)))

    temporary: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for target, image_format in zip(targets, ("PNG", "BMP")):
            fd, temp_name = tempfile.mkstemp(prefix=".image-convert-", suffix=target.suffix, dir=destination)
            os.close(fd)
            temp_path = Path(temp_name)
            temporary.append((temp_path, target))
            source_rgb.save(temp_path, format=image_format)

            with Image.open(temp_path) as converted:
                converted.load()
                converted_rgb = converted.convert("RGB")
                if converted_rgb.size != source_rgb.size or converted_rgb.tobytes() != source_rgb.tobytes():
                    raise RuntimeError(f"{image_format} 轉檔後像素與原 JPG 不一致，已中止。")

        # Publish only after both formats have passed exact decoded-pixel comparison.
        for temp_path, target in temporary:
            if target.exists():
                raise FileExistsError(f"輸出檔在轉檔期間出現，為避免覆寫已中止：{target}")
            temp_path.rename(target)
            published.append(target)
        return targets
    except Exception:
        for temp_path, _ in temporary:
            temp_path.unlink(missing_ok=True)
        for target in published:
            target.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="將 JPEG 轉成 PNG 與 BMP，並逐像素驗證。")
    parser.add_argument("input", type=Path, help="來源 JPG/JPEG 檔案")
    parser.add_argument("--output-dir", type=Path, help="輸出資料夾（預設為來源檔所在資料夾）")
    args = parser.parse_args()
    outputs = convert_jpeg(args.input, args.output_dir)
    for path in outputs:
        print(f"完成：{path}")
    print("驗證通過：兩個輸出檔解碼後的 RGB 像素均與來源 JPG 完全相同。")


if __name__ == "__main__":
    main()
