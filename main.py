from database import Database
import visualizer
from collections import Counter
from sklearn.model_selection import train_test_split

import numpy
import pywt
import cv2
import tensorflow as tf


def main():
    dataset_dir = "F://University/Final Project/dataset/Initial & repeat MRI in MS-Free Dataset"
    database = Database(dataset_dir)
    database.read_dataset()

    x, y = database.get_all_slices_with_labels()

    train_images, test_images, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=42)
    train_images, test_images, train_labels, test_labels = numpy.array(train_images), numpy.array(test_images), numpy.array(train_labels), numpy.array(test_labels)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(512, 512)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    prediction = numpy.argmax(model.predict(test_images), axis=-1)

    print('\n Test Accuracy: ', test_acc)
    get_evaluation_metrics(test_labels, prediction)


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
    normalized_image = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
    return normalized_image


def stationary_wavelet_transform(image):
    coefficients2 = pywt.swt2(image, 'bior1.3', 2)
    (LL1, (LH1, HL1, HH1)), (LL2, (LH2, HL2, HH2)) = coefficients2
    return [LL1, LH1, HL1, HL2, LL2, LH2, HL2, HH2]


# A function that returns accuracy, precision, recall and selectivity of the model based on test labels prediction
def get_evaluation_metrics(actual_labels, predicted_labels):
    tp = 0  # true positive
    fp = 0  # false positive
    tn = 0  # true negative
    fn = 0  # false negative

    # calculating values of fp, fn, tp and tn based on actual and predicted labels
    for i in range(len(actual_labels)):
        if actual_labels[i] == 1:
            if predicted_labels[i] == 1:
                tp += 1
            elif predicted_labels[i] == 0:
                fn += 1
        elif actual_labels[i] == 0:
            if predicted_labels[i] == 1:
                fp += 1
            elif predicted_labels[i] == 0:
                tn += 1

    # calculating evaluation metrics
    accuracy = divide_function(tp + tn, tp + tn + fp + fn)
    precision = divide_function(tp, tp + fp)
    recall = divide_function(tp, tp + fn)  # its also called sensitivity
    selectivity = divide_function(tn, tn + fp)

    print(
        "\naccuracy: {}\nprecision: {}\nrecall: {}\nselectivity: {}\n".format(accuracy, precision, recall, selectivity))

    return accuracy, precision, recall, selectivity


# A function that handles division by zero
def divide_function(numerator, denominator):
    try:
        value = numerator / denominator
    except ZeroDivisionError:
        value = float('Nan')

    return value


if __name__ == '__main__':
    main()
