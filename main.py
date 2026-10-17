"""Выравнивание изображения товара по масштабу и расположению образца.

По умолчанию программа использует `Исходники/1.webp` как образец и ищет ровно
одно исходное изображение в той же папке. Итоговое WEBP-изображение помещается
в папку `Результат`. Если у изображений прозрачный фон, используется альфа-канал;
иначе программа пытается определить сплошной фон по цветам на краях кадра.
"""

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
    """Возвращает координаты пикселей по периметру изображения.

    Args:
        size: Ширина и высота изображения в пикселях.

    Returns:
        Список координат `(x, y)` вдоль верхнего, нижнего, левого и правого
        краёв без повторного добавления углов.
    """
    width, height = size
    return (
        [(x, 0) for x in range(width)]
        + [(x, height - 1) for x in range(width)]
        + [(0, y) for y in range(1, height - 1)]
        + [(width - 1, y) for y in range(1, height - 1)]
    )


def _alpha_pixel(image: Image.Image, point: tuple[int, int]) -> int:
    """Читает значение пикселя одноканального изображения.

    Args:
        image: Одноканальное изображение, например альфа-маска.
        point: Координаты пикселя в виде `(x, y)`.

    Returns:
        Значение канала пикселя от 0 до 255.

    Raises:
        ValueError: Если пиксель не является целым числом, то есть изображение
            не одноканальное.
    """
    pixel = image.getpixel(point)
    if not isinstance(pixel, int):
        raise ValueError("Ожидалось одноканальное изображение.")
    return pixel


def _rgb_pixel(image: Image.Image, point: tuple[int, int]) -> tuple:
    """Читает RGB-пиксель и возвращает его цвет как кортеж каналов.

    Args:
        image: Изображение в режиме RGB.
        point: Координаты пикселя в виде `(x, y)`.

    Returns:
        Кортеж `(красный, зелёный, синий)`, значения каналов от 0 до 255.

    Raises:
        ValueError: Если пиксель не содержит ровно три цветовых канала.
    """
    pixel = image.getpixel(point)
    if not isinstance(pixel, tuple) or len(pixel) != 3:
        raise ValueError("Ожидалось RGB-изображение.")
    return pixel


def _has_transparent_background(image: Image.Image) -> bool:
    """Проверяет, прозрачно ли большинство краевых пикселей изображения.

    Args:
        image: Изображение в режиме RGBA.

    Returns:
        `True`, если более половины пикселей по периметру имеют альфа-значение
        не выше 16; иначе `False`.
    """
    alpha = image.getchannel("A")
    points = _border_points(image.size)
    
    # Частично прозрачные пиксели считаются фоном, чтобы учитывать сглаженные
    # края изображения, сохранённого с альфа-каналом.
    transparent = sum(_alpha_pixel(alpha, point) <= 16 for point in points)
    return transparent > len(points) // 2


def _solid_background_color(image: Image.Image) -> tuple[int, int, int]:
    """Оценивает цвет однотонного фона по наиболее частому цвету на краях.

    Args:
        image: Изображение, в котором нужно оценить фон.

    Returns:
        RGB-цвет фона. Перед подсчётом каналы объединяются в корзины по 16
        уровней, а итоговый цвет вычисляется как медиана подходящих пикселей.
    """
    rgb = image.convert("RGB")
    border = [_rgb_pixel(rgb, point) for point in _border_points(rgb.size)]
    # Корзины сглаживают небольшие колебания цвета, которые часто появляются
    # при сжатии изображения с потерями.
    dominant_bin, _ = Counter(
        (red // 16, green // 16, blue // 16) for red, green, blue in border
    ).most_common(1)[0]
    background_samples = [
        pixel
        for pixel in border
        if tuple(channel // 16 for channel in pixel) == dominant_bin
    ]

    def middle_channel(channel: int) -> int:
        """Возвращает медианное значение выбранного цветового канала.

        Args:
            channel: Индекс цветового канала: 0 для красного, 1 для зелёного
                или 2 для синего.

        Returns:
            Медианное значение канала среди пикселей фона.
        """
        values = sorted(pixel[channel] for pixel in background_samples)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return round((values[middle - 1] + values[middle]) / 2)

    return middle_channel(0), middle_channel(1), middle_channel(2)


def _opaque_foreground_mask(
    image: Image.Image,
) -> tuple[Image.Image, tuple[int, int, int]]:
    """Создаёт маску объекта для изображения с непрозрачным фоном.

    Сначала оценивается цвет фона по периметру. Затем с близких по цвету
    краевых точек выполняется заливка связных пикселей; залитые участки
    становятся прозрачными в маске.

    Args:
        image: Исходное изображение; его RGB-версия используется для сравнения
            цветов, независимо от режима входного файла.

    Returns:
        Пара из одноканальной маски альфа-прозрачности и оценённого RGB-цвета
        фона. Белые пиксели маски остаются видимыми, чёрные скрываются.
    """
    background_color = _solid_background_color(image)
    rgb = image.convert("RGB")
    marked = rgb.convert("RGBA")
    alpha = marked.getchannel("A")
    points = _border_points(rgb.size)
    # Ограничиваем число начальных точек заливки примерно пятьюстами,
    # чтобы не запускать операцию для каждого пикселя периметра.
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
            # Заливаем только область, связанную с краем и близкую по цвету
            # к оценённому фону, не затрагивая такие же цвета внутри объекта.
            ImageDraw.floodfill(marked, point, (0, 0, 0, 0), thresh=40)

    return marked.getchannel("A"), background_color


def _foreground(
    image: Image.Image,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int] | None]:
    """Подготавливает изображение и маску объекта независимо от типа фона.

    Args:
        image: Открытое Pillow-изображение. Ориентация по EXIF учитывается
            до преобразования в RGBA.

    Returns:
        Кортеж из нормализованного RGBA-изображения, маски объекта и цвета
        непрозрачного фона. Для прозрачного фона последний элемент равен
        `None`.
    """
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    if _has_transparent_background(rgba):
        return rgba, rgba.getchannel("A"), None

    # Если край кадра не прозрачен, извлекаем область однотонного фона
    # заливкой от краёв и записываем её в альфа-канал.
    mask, background_color = _opaque_foreground_mask(rgba)
    rgba.putalpha(mask)
    return rgba, mask, background_color


def _foreground_bounds(mask: Image.Image) -> tuple[int, int, int, int]:
    """Находит прямоугольные границы видимой части объекта по маске.

    Args:
        mask: Одноканальная маска, в которой значения выше порога обозначают
            видимую часть объекта.

    Returns:
        Границы объекта `(левая, верхняя, правая, нижняя)`, где правая и нижняя
        координаты не включаются в прямоугольник.

    Raises:
        ValueError: Если маска не содержит пикселей выше порога.
    """
    # Бинаризация игнорирует почти прозрачные краевые пиксели и мелкие шумы.
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
    """Подгоняет изображение-источник к размерам и расположению объекта образца.

    Args:
        reference_path: Путь к файлу-образцу, задающему размер холста,
            размер объекта и его положение.
        source_path: Путь к изображению того же товара, которое нужно
            выровнять.
        output_path: Путь, куда сохранить итоговое изображение в формате WEBP.

    Returns:
        `None`. Функция записывает файл по пути `output_path`.

    Raises:
        FileNotFoundError: Если файл образца или источник не найден.
        OSError: Если изображение не удаётся открыть или результат не удаётся
            записать.
        ValueError: Если объект не удаётся обнаружить по маске.
    """
    # Копируем декодированные пиксели до закрытия файлов, открытых через with.
    with Image.open(reference_path) as reference_image:
        reference = ImageOps.exif_transpose(reference_image).convert("RGBA")
    with Image.open(source_path) as source_image:
        source = ImageOps.exif_transpose(source_image).convert("RGBA")

    reference_rgba, reference_mask, reference_background = _foreground(reference)
    source_rgba, source_mask, _ = _foreground(source)
    reference_bounds = _foreground_bounds(reference_mask)
    source_bounds = _foreground_bounds(source_mask)

    left, top, right, bottom = reference_bounds
    # Масштабируем только обрезанную область товара до размеров объекта
    # образца, затем размещаем её в том же прямоугольнике на холсте.
    target_size = (right - left, bottom - top)
    source_crop = source_rgba.crop(source_bounds)
    aligned = source_crop.resize(target_size, Image.Resampling.LANCZOS)

    # Сохраняем тип фона образца: прозрачность для прозрачного фона, иначе
    # заполняем холст найденным однотонным цветом.
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
    """Находит единственное изображение-источник рядом с образцом.

    Args:
        reference_path: Путь к образцу; поиск выполняется в той же папке.

    Returns:
        Путь к единственному подходящему файлу, кроме самого образца.

    Raises:
        ValueError: Если найдено ноль или больше одного подходящего файла.
    """
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
    """Ожидает нажатия клавиши перед выходом из программы.

    В Windows читается одна клавиша средствами `msvcrt`. В интерактивном
    Unix-терминале временно включается посимвольный ввод без ожидания Enter.
    Если ввод перенаправлен, используется стандартное ожидание Enter.

    Returns:
        `None`.

    Raises:
        RuntimeError: Если на Windows отсутствует функция чтения клавиши.
        EOFError: Если при перенаправленном вводе достигнут конец файла до
            нажатия Enter.
        OSError: Если системные операции чтения терминала завершаются ошибкой.
    """
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
        # Храним исходные параметры терминала и восстанавливаем их даже при
        # ошибке чтения или прерывании программы.
        tty.setraw(file_descriptor)
        print("Нажмите любую клавишу для выхода...", end="", flush=True)
        os.read(file_descriptor, 1)
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_settings)
    print()


def main() -> None:
    """Разбирает аргументы запуска, запускает обработку и ждёт выхода.

    Параметры берутся из командной строки: путь к образцу, исходному
    изображению и необязательный путь результата. Если источник не задан,
    он автоматически выбирается в папке образца.

    Args:
        Нет. Аргументы программы читаются из `sys.argv`.

    Returns:
        `None`. При некорректных путях или числе исходников argparse выводит
        сообщение и завершает работу с ненулевым кодом.
    """
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

    # Проверяем образец до поиска исходника, чтобы сообщить о проблеме с
    # обязательным входным файлом непосредственно.
    if not args.reference.is_file():
        parser.error(f"Файл-образец не найден: {args.reference}")
    if args.source is None:
        try:
            source_path = _find_source(args.reference)
        except ValueError as error:
            # Преобразуем внутреннюю ошибку выбора файла в штатное сообщение
            # командной строки без traceback.
            parser.error(str(error))
    else:
        source_path = args.source
    if not source_path.is_file():
        parser.error(f"Исходное изображение не найдено: {source_path}")
    if source_path.resolve() == args.reference.resolve():
        parser.error("Исходное изображение и образец не должны совпадать.")

    # Если путь результата не задан, сохраняем WEBP с исходным именем
    # в стандартную папку результата.
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{source_path.stem}.webp"
    align_image(args.reference, source_path, output_path)
    print(f"Результат сохранён")
    _wait_for_keypress()


if __name__ == "__main__":
    # Не запускаем обработку при импорте этого файла как модуля.
    main()
