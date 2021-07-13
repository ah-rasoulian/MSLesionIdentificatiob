import numpy
import cv2

kernel21 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                        ], numpy.uint8)

kernel17 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
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
                        ], numpy.uint8)

kernel13 = numpy.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
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
                        ], numpy.uint8)

kernel11 = numpy.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
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
                        ], numpy.uint8)

kernel9 = numpy.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 1, 1, 1, 0, 0, 0],
                       ], numpy.uint8)


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
    normalized_image = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
    return normalized_image


#########################################################################
# A function for skull stripping based on the following paper:
# S. Roy and P. Maji, “A simple skull stripping algorithm for brain MRI,” in 2015 Eighth International Conference on Advances in Pattern Recognition (ICAPR), Jan. 2015, pp. 1–6. doi: 10.1109/ICAPR.2015.7050671.
#########################################################################
def skull_stripping_1(slice_with_skull):
    # 1- Apply the median filter with window of size 3*3 to the input image
    denoised = cv2.medianBlur(slice_with_skull, 3)
    # 2- Compute the initial mean intensity value Ti of the image.
    initial_mean_intensity = numpy.mean(denoised)

    # 3- Identify the top, bottom, left, and right pixel locations, from where brain skull starts in the image,
    # considering gray values of the skull are greater than Ti.
    mask_higher_than_initial_mean = cv2.inRange(denoised, initial_mean_intensity, 255)
    contours, hierarchy = cv2.findContours(mask_higher_than_initial_mean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)

    # 4- Form a rectangle using the top, bottom, left, and right pixel locations.
    x, y, w, h = cv2.boundingRect(biggest_contour)

    # 5- Compute the final mean value Tf of the brain using the pixels located within the rectangle.
    final_mean_intensity = numpy.mean(denoised[y:y + h, x:x + w])
    mask_between_initial_and_final_mean = cv2.inRange(denoised, initial_mean_intensity, final_mean_intensity)

    # 6- Approximate the region of brain membrane or meninges that envelop the brain,
    # based on the assumption that the intensity of skull is more than Tf
    # and that of membrane is less than Tf .
    membrane_region = cv2.bitwise_and(denoised, denoised, mask=mask_between_initial_and_final_mean).astype(numpy.float)
    membrane_region[membrane_region == 0] = numpy.nan

    # 7- Set the average intensity value of membrane as the threshold value T.
    membrane_mean_intensity = numpy.nanmean(membrane_region)

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
    ret, thresh = cv2.threshold(dilated, 0.185*255, 255, cv2.THRESH_BINARY)

    # Transform Binary image I4 to unit 8 format (I5)
    skull = cv2.bitwise_and(slice_with_skull, slice_with_skull, mask=thresh)

    # Subtract (I6) I5 from I1
    brain_out = slice_with_skull - skull

    return brain_out
