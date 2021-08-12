import numpy as np
import cv2
import tensorflow_addons as tfa
import tensorflow as tf
from scipy.ndimage import zoom
import random

kernel21 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     ], np.uint8)

kernel17 = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     ], np.uint8)

kernel13 = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     ], np.uint8)

kernel11 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     ], np.uint8)

kernel9 = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    ], np.uint8)


class Patch:
    def __init__(self, x_in_original_image, y_in_original_image, patch_image):
        self.x_in_original_image = x_in_original_image
        self.y_in_original_image = y_in_original_image
        self.patch_image = patch_image

    def get_top_left_x(self):
        return self.x_in_original_image

    def get_top_left_y(self):
        return self.y_in_original_image

    def get_patch_image(self):
        return self.patch_image


def get_image_patches(original_image, patch_width, patch_height, horizontal_gap=1, vertical_gap=1):
    patches = []
    for y in range(0, original_image.shape[0], vertical_gap):
        if original_image.shape[0] - y < patch_height:
            break
        for x in range(0, original_image.shape[1], horizontal_gap):
            if original_image.shape[1] - x < patch_width:
                break
            patches.append(Patch(x, y, original_image[y:y + patch_height, x:x + patch_width]))
    return patches


def histogram_stretching(image):
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return normalized_image


#########################################################################
# A function for skull stripping based on the following paper:
# S. Roy and P. Maji, “A simple skull stripping algorithm for brain MRI,” in 2015 Eighth International Conference on Advances in Pattern Recognition (ICAPR), Jan. 2015, pp. 1–6. doi: 10.1109/ICAPR.2015.7050671.
#########################################################################
def skull_stripping_1(slice_with_skull):
    # 1- Apply the median filter with window of size 3*3 to the input image
    denoised = cv2.medianBlur(slice_with_skull, 3)
    # 2- Compute the initial mean intensity value Ti of the image.
    initial_mean_intensity = np.mean(denoised)

    # 3- Identify the top, bottom, left, and right pixel locations, from where brain skull starts in the image,
    # considering gray values of the skull are greater than Ti.
    mask_higher_than_initial_mean = cv2.inRange(denoised, initial_mean_intensity, 255)
    contours, hierarchy = cv2.findContours(mask_higher_than_initial_mean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    # 4- Form a rectangle using the top, bottom, left, and right pixel locations.
    x, y, w, h = cv2.boundingRect(biggest_contour)

    # 5- Compute the final mean value Tf of the brain using the pixels located within the rectangle.
    final_mean_intensity = np.mean(denoised[y:y + h, x:x + w])
    mask_between_initial_and_final_mean = cv2.inRange(denoised, initial_mean_intensity, final_mean_intensity)

    # 6- Approximate the region of brain membrane or meninges that envelop the brain,
    # based on the assumption that the intensity of skull is more than Tf
    # and that of membrane is less than Tf .
    membrane_region = cv2.bitwise_and(denoised, denoised, mask=mask_between_initial_and_final_mean).astype(np.float)
    membrane_region[membrane_region == 0] = np.nan

    # 7- Set the average intensity value of membrane as the threshold value T.
    membrane_mean_intensity = np.nanmean(membrane_region)

    # 8- Convert the given input image into binary image using the threshold T.
    ret, thresh = cv2.threshold(denoised, membrane_mean_intensity, 255, cv2.THRESH_BINARY)

    # 9- Apply a 13×13 opening morphological operation to the binary image in order to separate the skull from
    # the brain completely.
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel13)

    # 10- Find the largest connected component and consider it as brain.
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = opening.copy()
    img2[output != max_label] = 0

    # Finally, apply a 21×21 closing morphological operation to fill the gaps within and along the periphery
    # of the intracranial region.
    closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel21, iterations=2)

    # Get the parts of original image corresponding to the final mask computed
    brain_out = slice_with_skull.copy()
    brain_out[closing == 0] = 0
    return brain_out


#########################################################################
# A function for skull stripping based on the following paper:
# A. S. Bhadauria, V. Bhateja, M. Nigam, and A. Arya, “Skull Stripping of Brain MRI Using Mathematical Morphology,” in Smart Intelligent Computing and Applications, Singapore, 2020, pp. 775–780. doi: 10.1007/978-981-13-9282-5_75.
#########################################################################
def skull_stripping_2(slice_with_skull):
    # Perform Erosion (I2) operation on I1 using se, disk-shaped structuring element (x) of size 4
    eroded = cv2.morphologyEx(slice_with_skull, cv2.MORPH_ERODE, kernel9)

    # Perform Dilation (I3) of the eroded image I2 using the same se
    dilated = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel9)

    # Convert Dilated image I3 to binary format (I4) using 0.185 threshold
    ret, thresh = cv2.threshold(dilated, 0.185 * 255, 255, cv2.THRESH_BINARY)

    # Transform Binary image I4 to unit 8 format (I5)
    skull = cv2.bitwise_and(slice_with_skull, slice_with_skull, mask=thresh)

    # Subtract (I6) I5 from I1
    brain_out = slice_with_skull - skull

    return brain_out


#########################################################################
# A function for skull stripping based on the following paper:
# R. Roslan, N. Jamil, and R. Mahmud, “Skull stripping of MRI brain images using mathematical morphology,” in 2010 IEEE EMBS Conference on Biomedical Engineering and Sciences (IECBES), Nov. 2010, pp. 26–31. doi: 10.1109/IECBES.2010.5742193.
# Using its double thresholding method
#########################################################################
def skull_stripping_3(slice_with_skull):
    # 1- Apply double thresholding whereas values below 0.2 and above 0.7 will be mapped to zero, and others to 1
    ret, thresh = cv2.threshold(slice_with_skull, 0.7 * 255, 255, cv2.THRESH_TOZERO_INV)
    ret, thresh = cv2.threshold(thresh, 0.2 * 255, 255, cv2.THRESH_BINARY)

    # 2- Erode the binary image using kernel of disk-shaped with size of 10
    eroded = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel21)

    # 3- Dilate the eroded image using the same kernel
    dilated = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel21)

    # 4- Morphological enhancement using region filling
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel21, iterations=2)

    # Get the parts of original image corresponding to the final mask computed
    brain_out = slice_with_skull.copy()
    brain_out[closing == 0] = 0
    return brain_out


#########################################################################
# A function for skull stripping based on the following paper:
# R. Roslan, N. Jamil, and R. Mahmud, “Skull stripping of MRI brain images using mathematical morphology,” in 2010 IEEE EMBS Conference on Biomedical Engineering and Sciences (IECBES), Nov. 2010, pp. 26–31. doi: 10.1109/IECBES.2010.5742193.
# Using its Otsu thresholding method
#########################################################################
def skull_stripping_4(slice_with_skull):
    # 1- Apply Otsu thresholding
    ret, thresh = cv2.threshold(slice_with_skull, 0, 255, cv2.THRESH_OTSU)

    # 2- Erode the binary image using kernel of disk-shaped with size of 10
    eroded = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel21)

    # 3- Dilate the eroded image using the same kernel
    dilated = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel21)

    # 4- Morphological enhancement using region filling
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel21, iterations=2)

    # Get the parts of original image corresponding to the final mask computed
    brain_out = slice_with_skull.copy()
    brain_out[closing == 0] = 0
    return brain_out


###############################################################################################
# function that returns least sized image encompassing the brain with offset for possible error
###############################################################################################
def get_least_sized_image_encompassing_brain(original_slice, offset):
    # 1- Apply the median filter with window of size 21*21 to the input image
    denoised = cv2.medianBlur(original_slice, 21)

    # 2- Compute the mask
    ret, thresh = cv2.threshold(denoised, np.mean(denoised), 255, cv2.THRESH_BINARY)

    # 3- Get all contours according to the mask
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4- Find the coordination of least sized image encompassing the whole brain
    min_x, min_y, max_x, max_y = original_slice.shape[1], original_slice.shape[0], 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x, min_y, max_x, max_y = min(x, min_x), min(y, min_y), max(x + w, max_x), max(y + h, max_y)

    # 5- Add offset to measured dimensions considering possible errors
    min_x, min_y, max_x, max_y = max(min_x - offset, 0), max(min_y - offset, 0), min(max_x + offset,
                                                                                     original_slice.shape[1]), min(
        max_y + offset, original_slice.shape[0])

    # 6- return the portion of original image according to calculated dimensions and top left corner location and make it squared
    rect = original_slice[min_y:max_y, min_x:max_x]
    height, width = rect.shape
    max_dimension = max(height, width)
    squared = np.zeros((max_dimension, max_dimension), np.uint8)
    squared[int((max_dimension - height) / 2):int(max_dimension - (max_dimension - height) / 2), int((max_dimension - width) / 2):int(max_dimension - (max_dimension - width) / 2)] = rect
    return squared, min_x - (int(max_dimension - width) / 2), min_y - int((max_dimension - height) / 2)


def pre_processing(original_slice):
    # no_skull = skull_stripping_1(original_slice)
    return get_least_sized_image_encompassing_brain(original_slice, 16)
    # return original_slice, 0, 0


def image_rotation(image, angle):
    return tfa.image.rotate(image, angle).numpy()


def image_gamma_correction(image, gamma, gain=1):
    return tf.image.adjust_gamma(image, gamma, gain).numpy()


def image_gaussian_noise_injection(image, mean, variance):
    row, col = image.shape
    sigma = variance ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image / 255 + gauss
    return noisy


def image_translation(image, width_shift, height_shift):
    return tfa.image.translate(image, (width_shift, height_shift)).numpy()


# function copied from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
# it has a bug in
def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = max(((out.shape[0] - h) // 2), 0)
        trim_left = max(((out.shape[1] - w) // 2), 0)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def random_augment(image):
    actual_image = image[:, :, 0]
    random_preprocessing = random.randint(0, 2)
    # gamma correction
    if random_preprocessing == 0:
        gamma = random.uniform(0.8, 1.1)
        result = image_gamma_correction(actual_image, gamma)
    # noise injection
    elif random_preprocessing == 1:
        result = image_gaussian_noise_injection(actual_image, 0, 0.01) * 255
    # no preprocessing
    else:
        result = actual_image

    return np.expand_dims(result, 2)
