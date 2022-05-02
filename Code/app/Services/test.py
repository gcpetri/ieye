import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from app.Services.get_mask import GetMask

def get_test_image(image_index=1):

    inputDir = 'app/Images/'

    # eye image number to read
    N = image_index

    # read in the image
    image = np.array(cv2.imread(os.path.join(inputDir, 'eye_' + str(N) + '.jpg'), cv2.IMREAD_GRAYSCALE))

    # read in the mask or make one
    mask = None
    if os.path.exists(os.path.join(inputDir,'mask_' + str(N) + '.jpg')):
        mask = plt.imread(os.path.join(inputDir,'mask_' + str(N) + '.jpg'))
    else:
        mask = GetMask(image)
        plt.imshow(mask)
        plt.imsave(os.path.join(inputDir,'mask_' + str(N) + '.jpg'), mask)

    # normalizing the mask
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    info = np.iinfo(mask.dtype)
    mask = mask.astype(np.float32) / info.max
    mask[mask < 0.5] = 0

    # cut out the image given the mask so we just have the eye
    image[mask==0] = 0
    image = image[~np.all(image == 0, axis=1)]
    image = image[:,~np.all(image == 0, axis=0)]

    return image