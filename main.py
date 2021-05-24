from database import Database
import visualizer
from collections import Counter
from sklearn.model_selection import train_test_split

import numpy as np
import pywt
import cv2
import tensorflow as tf


def main():
    dataset_dir = "/home/amirhossein/Data/University/Final Project/Work/dataset/Health Lab - University of Cyprus/Initial & repeat MRI in MS-Free Dataset"
    database = Database(dataset_dir)
    database.read_dataset()

    x, y = database.get_all_slices_with_labels()

    train_images, test_images, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=42)
    train_images, test_images, train_labels, test_labels = np.array(train_images), np.array(test_images), np.array(train_labels), np.array(test_labels)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(512, 512)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\n Test Accuracy: ', test_acc)


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
