from __future__ import annotations

import argparse
from collections import Counter
import importlib
import os
import platform
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE = BASE_DIR / "Исходники" / "1.webp"
DEFAULT_OUTPUT_DIR = BASE_DIR / "Результат"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FOREGROUND_THRESHOLD = 64


def _border_points(size: tuple[int, int]) -> list[tuple[int, int]]:
    width, height = size
    return (
        [(x, 0) for x in range(width)]
        + [(x, height - 1) for x in range(width)]
        + [(0, y) for y in range(1, height - 1)]
        + [(width - 1, y) for y in range(1, height - 1)]
    )


def _alpha_pixel(image: Image.Image, point: tuple[int, int]) -> int:
    pixel = image.getpixel(point)
    if not isinstance(pixel, int):
        raise ValueError("Ожидалось одноканальное изображение.")
    return pixel


def _rgb_pixel(image: Image.Image, point: tuple[int, int]) -> tuple[int, int, int]:
    pixel = image.getpixel(point)
    if not isinstance(pixel, tuple) or len(pixel) != 3:
        raise ValueError("Ожидалось RGB-изображение.")
    return pixel[0], pixel[1], pixel[2]


def _has_transparent_background(image: Image.Image) -> bool:
    alpha = image.getchannel("A")
    points = _border_points(image.size)
    transparent = sum(_alpha_pixel(alpha, point) <= 16 for point in points)
    return transparent > len(points) // 2


def _solid_background_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    border = [_rgb_pixel(rgb, point) for point in _border_points(rgb.size)]
    dominant_bin, _ = Counter(
        (red // 16, green // 16, blue // 16) for red, green, blue in border
    ).most_common(1)[0]
    background_samples = [
        pixel
        for pixel in border
        if tuple(channel // 16 for channel in pixel) == dominant_bin
    ]

    def middle_channel(channel: int) -> int:
        values = sorted(pixel[channel] for pixel in background_samples)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return round((values[middle - 1] + values[middle]) / 2)

    return middle_channel(0), middle_channel(1), middle_channel(2)


def _opaque_foreground_mask(
    image: Image.Image,
) -> tuple[Image.Image, tuple[int, int, int]]:
    background_color = _solid_background_color(image)
    rgb = image.convert("RGB")
    marked = rgb.convert("RGBA")
    alpha = marked.getchannel("A")
    points = _border_points(rgb.size)
    step = max(1, len(points) // 500)

    for point in points[::step]:
        if _alpha_pixel(alpha, point) == 0:
            continue
        pixel = _rgb_pixel(rgb, point)
        if (
            max(
                abs(pixel[channel] - background_color[channel])
                for channel in range(3)
            )
            <= 48
        ):
            ImageDraw.floodfill(marked, point, (0, 0, 0, 0), thresh=40)

    return marked.getchannel("A"), background_color


def _foreground(
    image: Image.Image,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int] | None]:
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    if _has_transparent_background(rgba):
        return rgba, rgba.getchannel("A"), None

    mask, background_color = _opaque_foreground_mask(rgba)
    rgba.putalpha(mask)
    return rgba, mask, background_color


def _foreground_bounds(mask: Image.Image) -> tuple[int, int, int, int]:
    lookup_table = [
        255 if value >= FOREGROUND_THRESHOLD else 0 for value in range(256)
    ]
    binary_mask = mask.point(lookup_table)
    bounds = binary_mask.getbbox()
    if bounds is None:
        raise ValueError("Не удалось определить объект на изображении.")
    return bounds


def align_image(
    reference_path: Path,
    source_path: Path,
    output_path: Path,
) -> None:
    with Image.open(reference_path) as reference_image:
        reference = ImageOps.exif_transpose(reference_image).convert("RGBA")
    with Image.open(source_path) as source_image:
        source = ImageOps.exif_transpose(source_image).convert("RGBA")

    reference_rgba, reference_mask, reference_background = _foreground(reference)
    source_rgba, source_mask, _ = _foreground(source)
    reference_bounds = _foreground_bounds(reference_mask)
    source_bounds = _foreground_bounds(source_mask)

    left, top, right, bottom = reference_bounds
    target_size = (right - left, bottom - top)
    source_crop = source_rgba.crop(source_bounds)
    aligned = source_crop.resize(target_size, Image.Resampling.LANCZOS)

    if reference_background is None:
        canvas = Image.new("RGBA", reference_rgba.size, (0, 0, 0, 0))
    else:
        canvas = Image.new(
            "RGBA",
            reference_rgba.size,
            (*reference_background, 255),
        )
    canvas.alpha_composite(aligned, (left, top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="WEBP", quality=95, method=6)


def _find_source(reference_path: Path) -> Path:
    candidates = sorted(
        path
        for path in reference_path.parent.iterdir()
        if path.is_file()
        and path != reference_path
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Ожидалось ровно одно изображение рядом с образцом "
            f"{reference_path.name}, найдено: {len(candidates)}."
        )
    return candidates[0]


def _wait_for_keypress() -> None:
    if platform.system() == "Windows":
        getch = getattr(importlib.import_module("msvcrt"), "getch", None)
        if not callable(getch):
            raise RuntimeError("В этой версии Windows недоступно чтение клавиши.")
        print("Нажмите любую клавишу для выхода...", end="", flush=True)
        getch()
        return

    if not sys.stdin.isatty():
        input("Нажмите Enter для выхода...")
        return

    import termios
    import tty

    file_descriptor = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setraw(file_descriptor)
        print("Нажмите любую клавишу для выхода...", end="", flush=True)
        os.read(file_descriptor, 1)
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_settings)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Выравнивает изображение товара по изображению-образцу."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Путь к изображению-образцу (по умолчанию: Исходники/1.webp).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Путь к изображению товара; по умолчанию ищется рядом с образцом.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Путь к результату; по умолчанию: Результат/<имя исходника>.webp.",
    )
    args = parser.parse_args()

    if not args.reference.is_file():
        parser.error(f"Файл-образец не найден: {args.reference}")
    source_path = args.source or _find_source(args.reference)
    if not source_path.is_file():
        parser.error(f"Исходное изображение не найдено: {source_path}")
    if source_path.resolve() == args.reference.resolve():
        parser.error("Исходное изображение и образец не должны совпадать.")

    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{source_path.stem}.webp"
    align_image(args.reference, source_path, output_path)
    print(f"Результат сохранён")
    _wait_for_keypress()


if __name__ == "__main__":
    main()
