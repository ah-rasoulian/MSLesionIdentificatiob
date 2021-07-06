import numpy
import cv2


class Patch:
    def __init__(self, x_in_original_image, y_in_original_image, patch_image):
        self.x_in_original_image = x_in_original_image
        self.y_in_original_image = y_in_original_image
        self.patch_image = patch_image

    def get_top_left_location_in_original_image(self):
        return self.x_in_original_image, self.y_in_original_image

    def get_patch_image(self):
        return self.patch_image


def histogram_stretching(image):
    normalized_image = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
    return normalized_image


def remove_skull(slice_with_skull):
    # cv2.imshow('original', slice_with_skull)
    # cv2.waitKey()

    # ret, thresh = cv2.threshold(src=slice_with_skull, thresh=95, maxval=255, type=cv2.THRESH_TOZERO)

    # denoised = cv2.fastNlMeansDenoising(thresh, h=30, templateWindowSize=7, searchWindowSize=21)

    ret, thresh2 = cv2.threshold(src=slice_with_skull, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh2, connectivity=8)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = slice_with_skull.copy()
    img2[output != max_label] = 0

    contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        img2 = cv2.fillPoly(img2, pts=[contour], color=(255, 0, 0))

    brain_out = slice_with_skull.copy()
    brain_out[img2 == 0] = 0

    # cv2.imshow('brain out', brain_out)
    # cv2.waitKey(0)

    return brain_out
