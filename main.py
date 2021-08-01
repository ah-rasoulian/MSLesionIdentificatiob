import numpy as np

from database import Database
import visualizer
import preprocessor
from collections import Counter
from sklearn.model_selection import train_test_split

import numpy
import matplotlib.pyplot as plt
import pywt
import cv2
import tensorflow as tf


def main():
    dataset_dir = "F://University/Final Project/dataset/Initial & repeat MRI in MS-Free Dataset"
    database = Database(dataset_dir)
    database.read_dataset()
    # database.add_new_sample("F://University/Final Project/dataset/Initial & repeat MRI in MS-Free Dataset/AA")
    #
    x, y = database.get_all_patches_with_labels(64, 64, 64, 64)
    print(len(x))
    k_fold_cross_validation(x, y, 10, deep_model_1)
    print(Counter(y))

    # for patient in database.get_samples():
    #     for sample in patient.get_examinations():
    #         for slice_mri in sample.get_slices()[8:]:
    #             cv2.imshow("test", slice_mri.slice_image)
    #             cv2.waitKey(0)
    #
    #             no_skull = preprocessor.skull_stripping_1(slice_mri.slice_image)
    #             x, z, c = preprocessor.get_least_sized_image_encompassing_brain(no_skull)
    #             cv2.imshow("no skull", x)
    #             cv2.waitKey(0)
    #
    # for patient in database.get_samples():
    #     for examination in patient.get_examinations():
    #         visualizer.show_brain_mri(examination)


def stationary_wavelet_entropy_and_decision_tree_model(x, y):
    from skimage.measure import shannon_entropy
    from sklearn.tree import DecisionTreeClassifier

    stationary_wavelet_entropy = []
    for img in x:
        sub_bands_entropy = []
        for swt in stationary_wavelet_transform(img):
            sub_bands_entropy.append(shannon_entropy(swt))
        stationary_wavelet_entropy.append(sub_bands_entropy)

    train_data, test_data, train_labels, test_labels = train_test_split(stationary_wavelet_entropy, y, test_size=0.2,
                                                                        random_state=42)
    classifier = DecisionTreeClassifier()
    classifier.fit(train_data, train_labels)

    p = classifier.predict(test_data)
    get_evaluation_metrics(test_labels, p)


def deep_model_1(train_images, validation_images, test_images, train_labels, validation_labels, test_labels):
    METRICS = [

        tf.keras.metrics.Precision(name='precision'),        # tf.keras.metrics.TruePositives(name='tp'),
        # tf.keras.metrics.FalsePositives(name='fp'),
        # tf.keras.metrics.TrueNegatives(name='tn'),
        # tf.keras.metrics.FalseNegatives(name='fn'),
        # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='recall'),
    ]

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.fit(train_images, train_labels, epochs=10, validation_data=(validation_images, validation_labels))

    print('\n Test Result: \n')
    model.evaluate(test_images, test_labels, verbose=2)

    prediction = numpy.argmax(model.predict(test_images), axis=-1)
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


def stationary_wavelet_transform(image):
    coefficients2 = pywt.swt2(data=image, wavelet='bior1.1', level=2, trim_approx=True)
    cA_2, (cH_2, cV_2, cD_2), (cH_1, cV_1, cD_1) = coefficients2
    return [cA_2, cH_2, cV_2, cD_2, cH_1, cV_1, cD_1]


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


# A function that divides dataset into k parts, 70% for training - 20% for validation and 10% for test, whereas these ratios can change
# Then loops over different folds and runs segmentation method on them.
# k 10 , i = 0 , i-0.7
def k_fold_cross_validation(x, y, k, model_name):
    TRAIN_SIZE = 0.7
    VALIDATION_SIZE = 0.2
    TEST_SIZE = 1 - TRAIN_SIZE - VALIDATION_SIZE
    portions_length = len(y) // k

    for i in range(1):  # 1 must be changed into k after model is completed
        # calculating train indices
        if i + TRAIN_SIZE * k <= k:
            if k - (i + TRAIN_SIZE * k) < 1:
                train_indices = np.arange(i * portions_length, len(y))
            else:
                train_indices = np.arange(i * portions_length, (i + TRAIN_SIZE * k) * portions_length)
        else:
            train_indices = np.concatenate(
                [np.arange(i * portions_length, len(y)), np.arange(0, ((i + TRAIN_SIZE * k) % k) * portions_length)])

        # calculating validation indices
        if i + (TRAIN_SIZE + VALIDATION_SIZE) * k <= k:
            if k - (i + (TRAIN_SIZE + VALIDATION_SIZE) * k) < 1:
                validation_indices = np.arange((i + TRAIN_SIZE * k) * portions_length, len(y))
            else:
                validation_indices = np.arange((i + TRAIN_SIZE * k) * portions_length,
                                               (i + (TRAIN_SIZE + VALIDATION_SIZE) * k) * portions_length)
        else:
            if i + TRAIN_SIZE * k <= k:
                validation_indices = np.concatenate([np.arange((i + TRAIN_SIZE * k) * portions_length, len(y)),
                                                     np.arange(0, ((i + (
                                                                 TRAIN_SIZE + VALIDATION_SIZE) * k) % k) * portions_length)])
            else:
                validation_indices = np.arange(((i + TRAIN_SIZE * k) % k) * portions_length,
                                               ((i + (TRAIN_SIZE + VALIDATION_SIZE) * k) % k) * portions_length)

        # calculating test indices
        if i + (TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE) * k <= k:
            if k - (i + (TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE) * k) < 1:
                test_indices = np.arange((i + (TRAIN_SIZE + VALIDATION_SIZE) * k) * portions_length, len(y))
            else:
                test_indices = np.arange((i + (TRAIN_SIZE + VALIDATION_SIZE) * k) * portions_length,
                                         (i + (TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE) * k))
        else:
            if i + (TRAIN_SIZE + VALIDATION_SIZE) * k <= k:
                test_indices = np.concatenate(
                    [np.arange((i + (TRAIN_SIZE + VALIDATION_SIZE) * k) * portions_length, len(y)),
                     np.arange(0, ((i + (TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE) * k) % k) * portions_length)])
            else:
                test_indices = np.arange(((i + (TRAIN_SIZE + VALIDATION_SIZE) * k) % k) * portions_length,
                                         ((i + (TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE) * k) % k) * portions_length)

        # creating datasets
        x = numpy.array(x)
        y = numpy.array(y)
        train_indices = train_indices.astype(np.intp)
        validation_indices = validation_indices.astype(np.intp)
        test_indices = test_indices.astype(np.intp)

        train_images, validation_images, test_images, train_labels, validation_labels, test_labels = x[train_indices], x[validation_indices], x[test_indices], y[train_indices], y[validation_indices], y[test_indices]
        train_images, validation_images, test_images = train_images / 255.0, validation_images / 255, test_images / 255.0

        train_images = train_images[..., tf.newaxis]
        validation_images = validation_images[..., tf.newaxis]
        test_images = test_images[..., tf.newaxis]

        model_name(train_images, validation_images, test_images, train_labels, validation_labels, test_labels)


# A function that handles division by zero
def divide_function(numerator, denominator):
    try:
        value = numerator / denominator
    except ZeroDivisionError:
        value = float('Nan')

    return value


if __name__ == '__main__':
    main()
