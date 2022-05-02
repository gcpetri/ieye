from scipy.ndimage import gaussian_filter

# Regular Gaussian Filter
def filter_gaussian(image, sigma=1.0):
    '''
    # rule of thumb = kernel w, h should be 6 times sigma
    # we want it the kernel to be odd so that the peak is centered
    sz = sigma * 3 if int(sigma * 6) % 2 != 0 else sigma * 3 - 0.5
    dist = np.linspace(-sz, sz, int(sz * 2))

    # create a grid for this kernel
    xx, yy = np.meshgrid(dist, dist)

    # calculate the point distances
    d = xx * xx + yy * yy

    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp( - (d / (2 * sigma**2)))
    kernel = kernel / np.sum(kernel)
    return convolve2d(image, kernel, mode='same')
    '''
    return gaussian_filter(image, sigma=sigma)