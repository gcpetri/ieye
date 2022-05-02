import cv2
from app.Services.contrast import increase_contrast
from app.Services.gaussian import filter_gaussian
import numpy as np
import cv2

def prepare_image(img):

    # convert raw data to np array
    nparr = np.fromstring(img.read(), np.uint8)

    # original image from byte array
    original_image = np.array(cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE))

    # resize the image so they're all consistent
    image = cv2.resize(original_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

    # smooth image to remove noise
    image = filter_gaussian(image, sigma=2)

    # increase contrast image to detect only pronounces features
    image = increase_contrast(image, contrast=64)

    return original_image, image

def test_prepare_image(image):

    # resize the image so they're all consistent
    image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

    # smooth image to remove noise
    image = filter_gaussian(image, sigma=2)

    # increase contrast image to detect only pronounces features
    image = increase_contrast(image, contrast=120)

    return image