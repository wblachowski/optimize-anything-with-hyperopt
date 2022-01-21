import cv2
import numpy as np


def resize(img, params):
    height = params["resize"]
    return cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))


def binarize(img, params):
    if params["binarize"][0]:
        args = params["binarize"][1]
        img = cv2.adaptiveThreshold(
            img, 255, args[0], cv2.THRESH_BINARY, args[1], args[2]
        )
    return img


def denoise(img, params):
    if params["denoise"][0]:
        args = params["denoise"][1]
        img = cv2.fastNlMeansDenoising(
            img,
            None,
            h=float(args[0]),
            templateWindowSize=args[1],
            searchWindowSize=args[2],
        )
    return img


def erode_and_dilate(img, params):
    if params["binarize"][0]:
        args = params["binarize"][-1]
        if args["erode"][0]:
            kernel_size = args["erode"][1]
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            img = cv2.erode(img, kernel)
        if args["dilate"][0]:
            kernel_size = args["dilate"][1]
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            img = cv2.dilate(img, kernel)
    return img
