import cv2

def increase_contrast(image, contrast=64):
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    return cv2.addWeighted(image, alpha_c, image, 0, gamma_c)