import cv2
import pywt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import math
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import preprocessor
import os


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
def k_fold_cross_validation(x, y, k, model_name, input_shape, train_batch_size, augment, weighted_class):
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

        train_model(train_images, validation_images, test_images, train_labels, validation_labels, test_labels,
                    model_name, input_shape, train_batch_size, augment, weighted_class)


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
        ax.plot(epochs, train_metric, color='b', label='train')
        ax.plot(epochs, val_metric, color='r', label='validation')
        ax.set(xlabel='epochs', ylabel=metric)
        ax.legend(loc='best')

    plt.show()


def train_model(train_images, validation_images, test_images, train_labels, validation_labels, test_labels, model_name,
                input_shape, train_batch_size, augment, weighted_class):
    train_images = np.array([cv2.resize(x, input_shape) for x in train_images])
    validation_images = np.array([cv2.resize(x, input_shape) for x in validation_images])
    test_images = np.array([cv2.resize(x, input_shape) for x in test_images])

    validation_images = np.expand_dims(validation_images, -1)
    test_images = np.expand_dims(test_images, -1)
    train_images = np.expand_dims(train_images, -1)

    if augment:
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=15,
                                           width_shift_range=10,
                                           height_shift_range=10,
                                           zoom_range=[0.7, 1.3],
                                           preprocessing_function=preprocessor.random_augment)
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_datagen.fit(train_images)
    train_generator = train_datagen.flow(train_images, train_labels, batch_size=train_batch_size, shuffle=True)
    validation_datagen.fit(validation_images)
    validation_generator = validation_datagen.flow(validation_images, validation_labels)

    # show_augmented_images(train_generator, 10)

    model = model_name(input_shape)
    if weighted_class:
        class_weights = {}
        for key, value in Counter(train_labels).items():
            class_weights[key] = len(train_labels) / value
    else:
        class_weights = None

    history = model.fit(train_generator, epochs=500, validation_data=validation_generator, class_weight=class_weights,
                        verbose=2)

    print('\n Test Result: \n')
    model.evaluate(test_images, test_labels, verbose=2)

    history_metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'prc', 'tp', 'fp', 'tn', 'fn']
    show_history(history, history_metrics)


def CNN_model_7_layers(input_shape):
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
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 1)),
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
    return model


# Convolutional Neural Network based on the paper:
# [1]Y.-D. Zhang, C. Pan, J. Sun, and C. Tang, “Multiple sclerosis identification by convolutional neural network with dropout and parametric ReLU,” Journal of Computational Science, vol. 28, pp. 1–10, Sep. 2018, doi: 10.1016/j.jocs.2018.07.003.
def CNN_model_10_layers(input_shape):
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
        tf.keras.layers.InputLayer(input_shape=(input_shape[0], input_shape[1], 1)),

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
        optimizer=RMSprop(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.summary()
    return model


def vgg_model(input_shape):
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

    x = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 1))
    x = tf.keras.layers.Concatenate()([x, x, x])
    vgg = tf.keras.applications.VGG16(input_tensor=x, include_top=False,
                                      input_shape=(input_shape[0], input_shape[1], 3))
    for layer in vgg.layers:
        layer.trainable = False

    x = tf.keras.layers.Flatten()(vgg.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(vgg.input, x)

    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.summary()
    return model


# produce 150 images per training image via scaling, noise addition, gamma correction, translation and rotation based on the paper:
# [1]Y.-D. Zhang, C. Pan, J. Sun, and C. Tang, “Multiple sclerosis identification by convolutional neural network with dropout and parametric ReLU,” Journal of Computational Science, vol. 28, pp. 1–10, Sep. 2018, doi: 10.1016/j.jocs.2018.07.003.
def augment_train_images(images, labels):
    parent_path = 'F:\\University\\Final Project\\dataset\\data-augmented\\'
    parent_dirs = os.listdir(parent_path)
    if len(parent_dirs) > 0:
        new_dir_name = str(int(parent_dirs[len(parent_dirs) - 1]) + 1)
    else:
        new_dir_name = '0'

    new_dir_path = os.path.join(parent_path, new_dir_name)
    os.mkdir(new_dir_path)
    class0_dir = os.path.join(new_dir_path, '0')
    os.mkdir(class0_dir)
    class1_dir = os.path.join(new_dir_path, '1')
    os.mkdir(class1_dir)

    image_number = -1
    for i, image in enumerate(images):
        # image rotation
        for angle in range(-15, 15, 1):
            image_number += 1
            angle_radiance = angle / 180 * np.pi
            if labels[i] == 0:
                cv2.imwrite(class0_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_rotation(image, angle_radiance))
            else:
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_rotation(image, angle_radiance))

        # gamma correction
        for gamma_value in range(70, 130, 2):
            image_number += 1
            if labels[i] == 0:
                cv2.imwrite(class0_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_gamma_correction(image, gamma_value / 100))
            else:
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_gamma_correction(image, gamma_value / 100))

        # gaussian noise injection with mean 0 and variance 0.01
        for j in range(30):
            image_number += 1
            if labels[i] == 0:
                cv2.imwrite(class0_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_gaussian_noise_injection(image, 0, 0.01) * 255)
            else:
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_gaussian_noise_injection(image, 0, 0.01) * 255)

        # random translation within 0-10 pixels
        for j in np.random.randint(-100, 100, 30):
            image_number += 1
            if j < 0:
                width_shift, height_shift = -(-i // 10), -(-i % 10)
            else:
                width_shift, height_shift = i // 10, i % 10

            if labels[i] == 0:
                cv2.imwrite(class0_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_translation(image, width_shift, height_shift))
            else:
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_translation(image, width_shift, height_shift))

        # zoom image
        for scale_factor in range(70, 130, 2):
            image_number += 1
            if labels[i] == 0:
                cv2.imwrite(class0_dir + '\\' + str(image_number) + '.png',
                            preprocessor.clipped_zoom(image, scale_factor / 100))
            else:
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.clipped_zoom(image, scale_factor / 100))

    return new_dir_path


def show_augmented_images(generator, k):
    for i in range(k):
        image, label = generator.next()
        cv2.imshow('augmented samples', image[0, :, :, 0])
        cv2.waitKey(0)
