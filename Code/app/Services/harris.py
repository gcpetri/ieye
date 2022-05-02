from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import cv2
from app.Services.db import eye_collection, np_to_kp, convert_img_np, convert_des_np

from app.Services.gaussian import filter_gaussian

# Sorbel Gradient Kernels (7x7)
def gradient_x(image):
    #kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_x = np.array([
        [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
        [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
        [-3/9, -2/4, -1/1, 0, 1/1, 2/4, 3/9],
        [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
        [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
        [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18]
    ])
    return convolve2d(image, kernel_x, mode='same')
def gradient_y(image):
    #kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_y = np.transpose(np.array([
        [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
        [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
        [-3/9, -2/4, -1/1, 0, 1/1, 2/4, 3/9],
        [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
        [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
        [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18]
    ]))
    return convolve2d(image, kernel_y, mode='same')

# my original implementation
def h_detect(image, k=0.05, thres=0.99):
    I_x = gradient_x(image)
    I_y = gradient_y(image)

    I_xx = filter_gaussian(I_x ** 2, sigma=1)
    I_xy = filter_gaussian(I_y * I_x, sigma=1)
    I_yy = filter_gaussian(I_y ** 2, sigma=1)

    # determinant
    detA = I_xx * I_yy - I_xy ** 2
    # trace
    traceA = I_xx + I_yy
    # response
    R = detA - k * traceA ** 2
    R[R < float(np.max(R) * thres)] = 0
    indices = []
    for i, value_i in enumerate(R):
        for j, value_j in enumerate(value_i):
            if value_j > 0:
                indices.append((i,j))
    print('corners founds: ', len(indices))
    return R, np.array(indices)
def plot_corners_and_edges(image, R, center, indices):
    img = np.copy(image)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #img[R > 0] = [255,0,0]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    ax.set_title("corners")
    for i,j in indices:
        ax.arrow(center[0], center[1], i - center[0], j - center[1],
            shape='full', color='r', length_includes_head=True, head_length=0.4, head_width=0.2)
    ax.imshow(img)
    plt.show()

# detector using cv2.cornerHarris
def h_detect_2(image, center, p=50):
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, blockSize=18, ksize=21, k=0.05)
    print(dst.shape)
    print(len(dst))
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    x = np.unravel_index(np.argsort(dst.ravel())[-len(dst):], dst.shape)

    # Threshold for an optimal value, it may vary depending on the image.
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img[x]=[0,0,255]

    plt.plot(center[0], center[1], "og", markersize=5)  # og:shorthand for green circle

    plt.imshow(img)
    plt.show()

    np.stack((x[0], x[1]), axis=-1)

    return x

def compute_heuristic(image, center, corners):
    # get the corner descriptors
    all_data = []
    for corner in corners:
        data = dict()
        data['desc'] = image[corner[0]-8:corner[0]+8,corner[1]-8:corner[1]+8]
        d = np.linalg.norm(corner - center, axis=0, keepdims=True)
        data['dist'] = d
        data['ang'] = np.arctan2(*(corner / d))
        all_data.append(data)
    
    return all_data

# sift detect 
# this gets the features and their descriptors (+ their orientation)
# so easy
def sift_detect(image):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    good_kp = []
    for key_point in kp:
        if key_point.size < 30 and key_point.size > 5:
            good_kp.append(key_point)
    img = cv2.drawKeypoints(image, good_kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img, kp, np.array(des)

# use FLANN matcher
def flann_match(image, kp, des):

    print('des', des)
    if not np.any(kp) or not np.any(des):
        print('new key points')
        return np.array([])

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=30)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # get all logged-in user eye data
    data = eye_collection.find()
    
    # match on each
    match_array = []
    for user_data in data:
        user_name = user_data['name']
        user_img = convert_img_np(user_data['img'])
        user_kp = np_to_kp(user_data['kp'])
        user_des = convert_des_np(user_data['des'], user_data['des-shape'])

        print('user_des', user_des)

        # match to all images in db
        matches = flann.knnMatch(des, user_des, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        num_matches = 0
        for i, (m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i] = [1,0]
                num_matches += 1
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
        match_img = cv2.drawMatchesKnn(image, kp, user_img, user_kp, matches, None, **draw_params)
        match_array.append([num_matches, user_name, match_img])
    return np.array(match_array)
