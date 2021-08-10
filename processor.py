import cv2
import pywt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import math
from collections import Counter
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def stationary_wavelet_transform(image):
    coefficients2 = pywt.swt2(data=image, wavelet='bior1.1', level=2, trim_approx=True)
    cA_2, (cH_2, cV_2, cD_2), (cH_1, cV_1, cD_1) = coefficients2
    return [cA_2, cH_2, cV_2, cD_2, cH_1, cV_1, cD_1]


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
        x = np.array(x)
        y = np.array(y)
        train_indices = train_indices.astype(np.intp)
        validation_indices = validation_indices.astype(np.intp)
        test_indices = test_indices.astype(np.intp)

        train_images, validation_images, test_images, train_labels, validation_labels, test_labels = x[train_indices], \
                                                                                                     x[
                                                                                                         validation_indices], \
                                                                                                     x[test_indices], y[
                                                                                                         train_indices], \
                                                                                                     y[
                                                                                                         validation_indices], \
                                                                                                     y[test_indices]

        model_name(train_images, validation_images, test_images, train_labels, validation_labels, test_labels)


def deep_model_patch(train_images, validation_images, test_images, train_labels, validation_labels, test_labels):
    train_images, validation_images, test_images = train_images / 255.0, validation_images / 255, test_images / 255.0

    train_images = train_images[..., tf.newaxis]
    validation_images = validation_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
    ]

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.summary()

    class_weights = {}
    for key, value in Counter(train_labels).items():
        class_weights[key] = len(train_labels) / value

    history = model.fit(train_images, train_labels, epochs=100, batch_size=2048, shuffle=True,
                        validation_data=(validation_images, validation_labels), validation_batch_size=1024,
                        class_weight=class_weights, verbose=2)

    print('\n Test Result: \n')
    model.evaluate(test_images, test_labels, verbose=2)

    prediction = np.argmax(model.predict(test_images), axis=-1)
    get_evaluation_metrics(test_labels, prediction)

    history_metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'prc', 'tp', 'fp', 'tn', 'fn']
    show_history(history, history_metrics)


def show_history(history, metrics):
    number_of_columns = math.floor(math.sqrt(len(metrics)))
    number_of_rows = math.ceil(len(metrics) / number_of_columns)

    fig = plt.figure(constrained_layout=True)
    grid = fig.add_gridspec(number_of_rows, number_of_columns)
    for i, metric in enumerate(metrics):
        train_metric = history.history[metric]
        val_metric = history.history['val_' + metric]
        epochs = np.arange(1, len(train_metric) + 1)

        ax = fig.add_subplot(
            grid[i % max(number_of_rows, number_of_columns), i // max(number_of_rows, number_of_columns)])
        ax.plot(epochs, train_metric, color='b', label='training')
        ax.plot(epochs, val_metric, color='r', label='validation')
        ax.set(xlabel='epochs', ylabel=metric)

    plt.show()


def deep_model_slice(train_images, validation_images, test_images, train_labels, validation_labels, test_labels):
    train_images = np.array([cv2.resize(x, (256, 256)) for x in train_images])
    validation_images = np.array([cv2.resize(x, (256, 256)) for x in validation_images])
    test_images = np.array([cv2.resize(x, (256, 256)) for x in test_images])

    train_images = np.expand_dims(train_images, -1)
    validation_images = np.expand_dims(validation_images, -1)
    test_images = np.expand_dims(test_images, -1)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=15,
                                       zoom_range=0.3,
                                       fill_mode='nearest'
                                       )
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    TRAIN_BATCH_SIZE = 128
    train_datagen.fit(train_images)
    train_generator = train_datagen.flow(train_images, train_labels, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    validation_datagen.fit(validation_images)
    validation_generator = validation_datagen.flow(validation_images, validation_labels)

    model = get_CNN_model_1()
    class_weights = {}
    for key, value in Counter(train_labels).items():
        class_weights[key] = len(train_labels) / value

    history = model.fit(train_generator, steps_per_epoch=len(train_labels) // TRAIN_BATCH_SIZE, epochs=200,
                        validation_data=validation_generator, class_weight=class_weights,
                        verbose=2)

    print('\n Test Result: \n')
    model.evaluate(test_images, test_labels, verbose=2)

    prediction = np.argmax(model.predict(test_images), axis=-1)
    get_evaluation_metrics(test_labels, prediction)

    history_metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'prc', 'tp', 'fp', 'tn', 'fn']
    show_history(history, history_metrics)


# Convolutional Neural Network based on the paper:
# [1]Y.-D. Zhang, C. Pan, J. Sun, and C. Tang, “Multiple sclerosis identification by convolutional neural network with dropout and parametric ReLU,” Journal of Computational Science, vol. 28, pp. 1–10, Sep. 2018, doi: 10.1016/j.jocs.2018.07.003.
def get_CNN_model_1():
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
    ]

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(256, 256, 1)),

        tf.keras.layers.ZeroPadding2D((2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5), strides=3, name='Conv_1'),
        tf.keras.layers.PReLU(tf.keras.initializers.Constant(0.25)),
        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=1, name='Pool_1'),

        tf.keras.layers.ZeroPadding2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), strides=3, name='Conv_2'),
        tf.keras.layers.PReLU(tf.keras.initializers.Constant(0.25)),
        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=1, name='Pool_2'),


        tf.keras.layers.Conv2D(32, (3, 3), strides=3, name='Conv_3'),
        tf.keras.layers.PReLU(tf.keras.initializers.Constant(0.25)),
        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=1, name='Pool_3'),

        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), strides=3, name='Conv_4'),
        tf.keras.layers.PReLU(tf.keras.initializers.Constant(0.25)),
        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=1, name='Pool_4'),

        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), strides=1, name='Conv_5'),
        tf.keras.layers.PReLU(tf.keras.initializers.Constant(0.25)),
        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=1, name='Pool_5'),

        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), strides=1, name='Conv_6'),
        tf.keras.layers.PReLU(tf.keras.initializers.Constant(0.25)),
        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=1, name='Pool_6'),

        tf.keras.layers.Conv2D(128, (1, 1), strides=1, name='Conv_7'),
        tf.keras.layers.PReLU(tf.keras.initializers.Constant(0.25)),
        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=1, name='Pool_7'),

        tf.keras.layers.Dropout(0.4, name='Dropout_1'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, name='FC_1'),

        tf.keras.layers.Dropout(0.5, name='Dropout_2'),
        tf.keras.layers.Dense(100, name='FC_2'),

        tf.keras.layers.Dropout(0.5, name='Dropout_3'),
        tf.keras.layers.Dense(1, name='FC_3', activation='sigmoid'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=5e-6),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.summary()
    return model
