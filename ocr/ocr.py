import re
from glob import glob
from pathlib import Path

import cv2
import pytesseract
from hyperopt import STATUS_OK
from Levenshtein import ratio

from plotting import display_ocr_result
from preprocessing import binarize, denoise, erode_and_dilate, resize


def cummulative_dissimilarity(params, directory="data", display=False):
    total_loss = 0
    images_dir = Path(directory) / "images"
    texts_dir = Path(directory) / "texts"
    for image in glob(str(images_dir / "*.jpg")):
        text = texts_dir / (Path(image).stem + ".txt")
        total_loss += dissimilarity(image, text, params, display=display)

    return {"loss": total_loss, "status": STATUS_OK, "params": params}


def dissimilarity(img_path, text_path, params, display=False):
    ocr_text, preprocessed_img = _ocr(img_path, params)
    with open(text_path) as file:
        expected_text = re.sub("\s+", " ", file.read().strip())
    similarity = ratio(re.sub("\s+", " ", ocr_text), expected_text)
    if display:
        display_ocr_result(ocr_text, preprocessed_img, similarity)
    return 1 - similarity


def _ocr(img_path, params):
    # Reading the image + converting to grayscale:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parametrized transformations:
    img = resize(img, params)
    img = binarize(img, params)
    img = denoise(img, params)
    img = erode_and_dilate(img, params)

    # OCR:
    text = pytesseract.image_to_string(img, config="--psm 6").strip()
    return text, img
