from database import Database
import visualizer
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2


def main():
    dataset_dir = "/home/amirhossein/Data/University/Final Project/Work/dataset/Health Lab - University of Cyprus/Initial & repeat MRI in MS-Free Dataset"
    database = Database(dataset_dir)
    database.read_dataset()

    x, y = database.get_all_slices_with_labels()

    image = x[12]
    normalized_image = image_normalization_using_histogram_stretching(image)

    cv2.imshow('mri', image)
    cv2.waitKey(0)

    cv2.imshow('mri', normalized_image)
    cv2.waitKey(0)


def print_labels_per_sample(database):
    for patient in database.get_samples():
        for brian_mri in patient.get_examinations():
            labels = []
            for slice_mri in brian_mri.get_slices():
                labels.append(slice_mri.does_contain_lesion())
            print(patient.patient_code, Counter(labels))


def visualize_sample_brain(database, sample_index, examination_number):
    patient = database.get_samples()[sample_index]
    visualizer.show_brain_mri(patient.get_examinations()[examination_number])


def image_normalization_using_histogram_stretching(image):
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return normalized_image


def stationary_wavelet_transform(image):
    coefficients2 = pywt.swt2(image, 'bior1.3', 2)
    (LL1, (LH1, HL1, HH1)), (LL2, (LH2, HL2, HH2)) = coefficients2
    return [LL1, LH1, HL1, HL2, LL2, LH2, HL2, HH2]


if __name__ == '__main__':
    main()
