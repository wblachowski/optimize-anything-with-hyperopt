import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def display_ocr_result(text, img, similarity):
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.title("Preprocessed image")
    plt.imshow(img, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)

    gap = 10

    lines = [line for line in text.split("\n") if line.strip()]
    longest_line = max(lines, key=len) if len(lines) else ""
    pil_image = Image.fromarray(np.full_like(img, 255))
    draw = ImageDraw.Draw(pil_image)
    font_size = _get_optimal_font_size(
        longest_line,
        len(lines),
        "arial.ttf",
        pil_image.size[0] - 2 * gap,
        pil_image.size[1] - 2 * gap,
        encoding="utf-8",
    )
    font_text = ImageFont.truetype(font="arial.ttf", size=font_size, encoding="utf-8")
    draw.text((gap, gap), text, 0, font=font_text)
    text_img = np.asarray(pil_image)
    plt.title(f"OCR result (similarity: {similarity:.4f})")
    plt.imshow(text_img, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()


def _get_optimal_font_size(text, num_lines, font, width, height, encoding):
    for size in reversed(range(0, 60, 1)):
        font_text = ImageFont.truetype(font=font, size=size, encoding=encoding)
        text_size = font_text.getsize(text)
        if text_size[0] <= width and text_size[1] * num_lines <= height:
            return size
    return 1
