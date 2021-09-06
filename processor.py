import cv2
import pywt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import math
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import preprocessor
import os
import random
from database import Database

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
def k_fold_cross_validation(x, y, k, model_name, input_shape, output_dim, train_batch_size, augment_type,
                            weighted_class,
                            fine_tune, num_epochs, manual_augment_path=None):
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
                    model_name, input_shape, output_dim, train_batch_size, augment_type, weighted_class, fine_tune,
                    num_epochs, manual_augment_path)


def show_history(history, metrics, fine_tune_history):
    number_of_columns = math.floor(math.sqrt(len(metrics)))
    number_of_rows = math.ceil(len(metrics) / number_of_columns)

    fig = plt.figure(constrained_layout=True)
    grid = fig.add_gridspec(number_of_rows, number_of_columns)
    for i, metric in enumerate(metrics):
        train_metric = history.history[metric]
        val_metric = history.history['val_' + metric]

        if fine_tune_history is not None:
            for new_history in fine_tune_history:
                train_metric.extend(new_history.history[metric])
                val_metric.extend(new_history.history['val_' + metric])

        epochs = np.arange(1, len(train_metric) + 1)

        ax = fig.add_subplot(
            grid[i % max(number_of_rows, number_of_columns), i // max(number_of_rows, number_of_columns)])
        ax.plot(epochs, train_metric, color='b', label='train')
        ax.plot(epochs, val_metric, color='r', label='validation')
        ax.set(xlabel='epochs', ylabel=metric)
        ax.legend(loc='best')

    plt.show()


def train_model(train_images, validation_images, test_images, train_labels, validation_labels, test_labels, model_name,
                input_shape, output_dim, train_batch_size, augment_type, weighted_class, fine_tune, num_epochs,
                manual_augment_path):  # augment_type: 0 for no augment, 1 for Image data generator, 2 for manual

    train_images = np.array([cv2.resize(x, input_shape) for x in train_images])
    validation_images = np.array([cv2.resize(x, input_shape) for x in validation_images])
    validation_images = np.expand_dims(validation_images, -1)

    if test_images is not None:
        test_images = np.array([cv2.resize(x, input_shape) for x in test_images])
        test_images = np.expand_dims(test_images, -1)

    if augment_type == 1:
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=15,
                                           width_shift_range=10,
                                           height_shift_range=10,
                                           zoom_range=[0.7, 1.3],
                                           preprocessing_function=preprocessor.random_augment)
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    if augment_type == 2 or augment_type == 3:
        if manual_augment_path is None:
            if augment_type == 2:
                manual_augment_path = manual_augmentation(train_images, train_labels)
            else:
                manual_augment_path = manual_balance_augmentation(train_images, train_labels)

        if output_dim == 2:
            class_mode = 'categorical'
        else:
            class_mode = 'binary'
        train_generator = train_datagen.flow_from_directory(directory=manual_augment_path,
                                                            target_size=input_shape, color_mode='grayscale',
                                                            class_mode=class_mode,
                                                            batch_size=train_batch_size,
                                                            shuffle=True)
        train_images = None
    else:
        train_images = np.expand_dims(train_images, -1)
        train_datagen.fit(train_images)
        if output_dim == 2:
            train_labels = tf.keras.utils.to_categorical(train_labels)
        train_generator = train_datagen.flow(train_images, train_labels, batch_size=train_batch_size, shuffle=True)
    validation_datagen.fit(validation_images)
    if output_dim == 2:
        validation_labels = tf.keras.utils.to_categorical(validation_labels)
    validation_generator = validation_datagen.flow(validation_images, validation_labels,
                                                   batch_size=max(1, train_batch_size // 4))

    # show_augmented_images(train_generator, 10)

    model = model_name(input_shape)
    if weighted_class:
        class_weights = {}
        for key, value in Counter(train_labels).items():
            class_weights[key] = len(train_labels) / value
    else:
        class_weights = None

    history = model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator,
                        class_weight=class_weights,
                        verbose=2)
    # fine tuning
    fine_tune_history = None
    if fine_tune:
        if test_labels is not None:
            if output_dim == 2:
                test_labels = tf.keras.utils.to_categorical(test_labels)
            for i in range(3):
                model.fit(train_generator, epochs=1, validation_data=validation_generator,
                          class_weight=class_weights,
                          verbose=1)

        model = fine_tuning(model)
        fine_tune_history = model.fit(train_generator, epochs=10, validation_data=validation_generator,
                                      class_weight=class_weights,
                                      verbose=2)

    if test_labels is not None:
        if output_dim == 2:
            test_labels = tf.keras.utils.to_categorical(test_labels)
        print('\n Test Result: \n')
        model.evaluate(test_images, test_labels, verbose=2)

    history_metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'prc', 'tp', 'fp', 'tn', 'fn']
    show_history(history, history_metrics, fine_tune_history)


def CNN_model_7_layers(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 1),
                               padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.summary()
    return model


# Convolutional Neural Network based on the paper:
# [1]Y.-D. Zhang, C. Pan, J. Sun, and C. Tang, “Multiple sclerosis identification by convolutional neural network with dropout and parametric ReLU,” Journal of Computational Science, vol. 28, pp. 1–10, Sep. 2018, doi: 10.1016/j.jocs.2018.07.003.
def CNN_model_10_layers(input_shape):
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
        loss=tf.keras.losses.binary_crossentropy(),
        metrics=METRICS)

    model.summary()
    return model


def vgg_model(input_shape):
    inputs = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 1))
    inputs = tf.keras.layers.Concatenate()([inputs, inputs, inputs])

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=inputs,
                                      input_shape=(input_shape[0], input_shape[1], 3))
    for layer in vgg.layers:
        layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(vgg.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(vgg.input, x)

    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.summary()
    return model


def fine_tuning(model, trainable_conv_layers):
    model.trainable = True
    last_trainable_layer_id = 0
    for i in range(len(model.layers) - 1, 0, -1):
        if model.layers[i].name.__contains__('conv'):
            trainable_conv_layers -= 1
        if trainable_conv_layers == 0:
            last_trainable_layer_id = i
            break

    for i in range(last_trainable_layer_id):
        model.layers[i].trainable = False

    model.compile(
        optimizer=RMSprop(learning_rate=5e-6),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)
    model.summary()
    return model


def resnet_model(input_shape):
    x = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 1))
    x = tf.keras.layers.Concatenate()([x, x, x])
    resnet = tf.keras.applications.ResNet50(input_tensor=x, include_top=False, weights='imagenet',
                                            input_shape=(input_shape[0], input_shape[1], 3))
    for layer in resnet.layers:
        layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(resnet.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(resnet.input, x)

    model.compile(
        optimizer=RMSprop(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.summary()
    return model


# produce 150 images per training image via scaling, noise addition, gamma correction, translation and rotation based on the paper:
# [1]Y.-D. Zhang, C. Pan, J. Sun, and C. Tang, “Multiple sclerosis identification by convolutional neural network with dropout and parametric ReLU,” Journal of Computational Science, vol. 28, pp. 1–10, Sep. 2018, doi: 10.1016/j.jocs.2018.07.003.
def manual_augmentation(images, labels):
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


def CNN_model_test(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape[0], input_shape[1], 1)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AvgPool2D((2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AvgPool2D((2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AvgPool2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.summary()
    return model


# copied from DeepLearning.AI Tensorflow course by Laurence Moroney
def show_model_pathway(model, sample):
    sample = np.expand_dims(sample, 0)

    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]

    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(sample)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # -----------------------------------------------------------------------
    # Now let's display our representations
    # -----------------------------------------------------------------------
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # -------------------------------------------
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            # -------------------------------------------
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)

            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # -------------------------------------------------
            # Postprocess the feature to be visually palatable
            # -------------------------------------------------
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size: (i + 1) * size] = x  # Tile each filter into a horizontal grid

            # -----------------
            # Display the grid
            # -----------------

            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


def manual_balance_augmentation(images, labels):
    print("test", Counter(labels))
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
        if labels[i] == 0:
            image_number += 1
            cv2.imwrite(class0_dir + '\\' + str(image_number) + '.png', image)
        else:
            image_number += 1
            cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png', image)

            # image rotation
            for angle in [-15, -10, 10, 15]:
                image_number += 1
                angle_radiance = angle / 180 * np.pi
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_rotation(image, angle_radiance))

            # gamma correction
            for gamma_value in [85, 90, 105, 110]:
                image_number += 1
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_gamma_correction(image, gamma_value / 100))

            # gaussian noise injection with mean 0 and variance 0.0025
            for j in range(3):
                image_number += 1
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_gaussian_noise_injection(image, 0, 0.0025) * 255)

            # random translation within 0-10 pixels
            for j in [(-5, -5), (-5, 0), (0, 5), (5, 5)]:
                image_number += 1
                width_shift, height_shift = j[0], j[1]
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.image_translation(image, width_shift, height_shift))

            # zoom image
            for scale_factor in [85, 90, 105, 110]:
                image_number += 1
                cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png',
                            preprocessor.clipped_zoom(image, scale_factor / 100))

    return new_dir_path


def train_manual_dataset(x, y, model_name, input_shape, train_batch_size, num_epochs, fine_tune, manual_path=None):
    if manual_path is None:
        reduced_x, reduced_y = [], []
        label_1_indexes = [i for i, val in enumerate(y) if val == 1]
        number_of_new_class_1_images = 20 * len(label_1_indexes)
        label_0_indexes = random.sample([i for i, val in enumerate(y) if val == 0], number_of_new_class_1_images)

        reduced_x.extend([x[i] for i in label_1_indexes])
        reduced_y.extend([y[i] for i in label_1_indexes])
        reduced_x.extend([x[i] for i in label_0_indexes])
        reduced_y.extend([y[i] for i in label_0_indexes])

        manual_path = manual_balance_augmentation(reduced_x, reduced_y)

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    validation_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    seed = random.randint(1, 100)
    train_generator = train_datagen.flow_from_directory(directory=manual_path,
                                                        target_size=input_shape, color_mode='grayscale',
                                                        class_mode='binary',
                                                        batch_size=train_batch_size,
                                                        shuffle=True,
                                                        subset='training',
                                                        seed=42)
    validation_generator = validation_datagen.flow_from_directory(directory=manual_path,
                                                                  target_size=input_shape, color_mode='grayscale',
                                                                  class_mode='binary',
                                                                  batch_size=train_batch_size // 4,
                                                                  shuffle=True,
                                                                  subset='validation',
                                                                  seed=42)

    model = model_name(input_shape)

    history = model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator,
                        verbose=1)
    # fine tuning
    fine_tune_history = None
    if fine_tune:
        model = fine_tuning(model)
        fine_tune_history = model.fit(train_generator, epochs=10, validation_data=validation_generator, verbose=1)

    print('\n original result \n')
    x, y = np.expand_dims(np.array(x), -1), np.expand_dims(np.array(y), -1)
    model.evaluate(x, y, verbose=1)

    history_metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'prc', 'tp', 'fp', 'tn', 'fn']
    show_history(history, history_metrics, fine_tune_history)


# based on the paper:
# [1]S.-H. Wang et al., “Multiple Sclerosis Identification by 14-Layer Convolutional Neural Network With Batch Normalization, Dropout, and Stochastic Pooling,” Frontiers in Neuroscience, vol. 12, p. 818, 2018, doi: 10.3389/fnins.2018.00818.
def CNN_model_14_layers(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape[0], input_shape[1], 1)),

        tf.keras.layers.Conv2D(8, (3, 3), padding='same', strides=2, name='Conv_1'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same', name='Pool_1'),

        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2, name='Conv_2'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same', name='Pool_2'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=1, name='Conv_3'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=1, name='Conv_4'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=1, name='Conv_5'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same', name='Pool_3'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=1, name='Conv_6'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=1, name='Conv_7'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=1, name='Conv_8'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv_9'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv_10'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='Conv_11'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same', name='Pool_4'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=1e-2),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    model.summary()
    return model


def hold_out_method(x, y, model_name, input_shape, output_dim, train_batch_size, augment_type, weighted_class,
                    fine_tune,
                    num_epochs, manual_augment_path=None):
    x, y = np.array(x), np.array(y)

    train_images, val_images, train_labels, val_labels = train_test_split(x, y, test_size=0.3)
    validation_images, test_images, validation_labels, test_labels = train_test_split(val_images, val_labels,
                                                                                      test_size=0.3)

    train_model(train_images, validation_images, test_images, train_labels, validation_labels, test_labels, model_name,
                input_shape, output_dim,
                train_batch_size, augment_type, weighted_class, fine_tune, num_epochs, manual_augment_path)


def create_new_dataset(x, y, label_1_per_label_0):
    reduced_x, reduced_y = [], []
    label_1_indexes = [i for i, val in enumerate(y) if val == 1]
    number_of_new_class_1_images = label_1_per_label_0 * len(label_1_indexes)
    label_0_indexes = random.sample([i for i, val in enumerate(y) if val == 0], number_of_new_class_1_images)

    reduced_x.extend([x[i] for i in label_1_indexes])
    reduced_y.extend([y[i] for i in label_1_indexes])
    reduced_x.extend([x[i] for i in label_0_indexes])
    reduced_y.extend([y[i] for i in label_0_indexes])

    parent_path = 'F:\\University\\Final Project\\dataset\\new_dataset\\'
    parent_dirs = os.listdir(parent_path)
    if len(parent_dirs) > 0:
        new_dir_name = str(int(parent_dirs[len(parent_dirs) - 1]) + 1)
    else:
        new_dir_name = '0'

    train_images, val_images, train_labels, val_labels = train_test_split(reduced_x, reduced_y, test_size=0.3,
                                                                          stratify=reduced_y)
    validation_images, test_images, validation_labels, test_labels = train_test_split(val_images, val_labels,
                                                                                      test_size=0.3,
                                                                                      stratify=val_labels)
    new_dir_path = os.path.join(parent_path, new_dir_name)
    os.mkdir(new_dir_path)

    train_dir = os.path.join(new_dir_path, 'train')
    os.mkdir(train_dir)
    save_images(train_images, train_labels, train_dir)

    validation_dir = os.path.join(new_dir_path, 'validation')
    os.mkdir(validation_dir)
    save_images(validation_images, validation_labels, validation_dir)

    test_dir = os.path.join(new_dir_path, 'test')
    os.mkdir(test_dir)
    save_images(test_images, test_labels, test_dir)


def save_images(images, labels, parent_path):
    class0_dir = os.path.join(parent_path, '0')
    os.mkdir(class0_dir)
    class1_dir = os.path.join(parent_path, '1')
    os.mkdir(class1_dir)
    image_number = -1
    for i, image in enumerate(images):
        if labels[i] == 0:
            image_number += 1
            cv2.imwrite(class0_dir + '\\' + str(image_number) + '.png', image)
        else:
            image_number += 1
            cv2.imwrite(class1_dir + '\\' + str(image_number) + '.png', image)


def train_new_dataset(parent_path, model_name, fine_tune, num_epochs, fine_tune_epochs, fine_tune_trainable_conv_layers,
                      input_shape, train_batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=15,
                                       width_shift_range=5,
                                       height_shift_range=5,
                                       zoom_range=[0.8, 1.2],
                                       preprocessing_function=preprocessor.random_augment
                                       )
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(directory=os.path.join(parent_path, 'train'),
                                                        color_mode='grayscale',
                                                        class_mode='binary',
                                                        target_size=input_shape,
                                                        batch_size=train_batch_size,
                                                        shuffle=True)

    valid_generator = valid_datagen.flow_from_directory(directory=os.path.join(parent_path, 'validation'),
                                                        color_mode='grayscale',
                                                        class_mode='binary',
                                                        target_size=input_shape,
                                                        batch_size=max(1, train_batch_size // 4))

    test_generator = test_datagen.flow_from_directory(directory=os.path.join(parent_path, 'test'),
                                                      color_mode='grayscale',
                                                      class_mode='binary',
                                                      target_size=input_shape,
                                                      batch_size=max(1, train_batch_size // 8))

    model = model_name(input_shape)

    history = model.fit(train_generator, epochs=num_epochs, validation_data=valid_generator, verbose=1)
    # fine tuning
    fine_tune_history = []
    if fine_tune:
        for i in range(1, fine_tune_trainable_conv_layers + 1):
            model = fine_tuning(model, i)
            fine_tune_history.append(
                model.fit(train_generator, epochs=fine_tune_epochs, validation_data=valid_generator,
                          verbose=1))

    print('\ntest result:\n')
    model.evaluate(test_generator, verbose=1)

    history_metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'prc', 'tp', 'fp', 'tn', 'fn']
    show_history(history, history_metrics, fine_tune_history)
