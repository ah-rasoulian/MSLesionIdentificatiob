from database import Database
import visualizer
import preprocessor
import processor

import numpy as np
from collections import Counter
import cv2


def main():
    # dataset_dir = "F://University/Final Project/dataset/Initial & repeat MRI in MS-Free Dataset"
    # dataset_dir = "F:\\University\\Final Project\\dataset\\data-augmented\\3"
    # database = Database(dataset_dir)

    # -----------------------------------------------------read dataset-----------------------------------------------------
    # database.read_dataset()
    # database.read_images()
    # database.add_new_sample("F://University/Final Project/dataset/refined_dataset/AA")
    # database.add_new_sample("F://University/Final Project/dataset/refined_dataset/ARK")
    # database.add_new_sample("F://University/Final Project/dataset/refined_dataset/AT")
    # database.add_new_sample("F://University/Final Project/dataset/refined_dataset/CHEM")
    # ----------------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------read x and y-----------------------------------------------------
    # x, y = database.get_all_patches_with_labels(32, 32, 16, 16)
    # x, y = database.get_all_slices_with_labels()
    # x, y = database.get_patches_of_affected_slices_with_labels(32, 32, 16, 16)
    # x, y = database.get_images_with_labels()
    # print(len(y))
    # print(Counter(y))
    # processor.create_new_dataset(x, y, 1)
    # processor.create_new_dataset(x, y, 2)
    # processor.create_new_dataset(x, y, 3)
    # ----------------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------train model---------------------------------------------------------
    processor.train_new_dataset(parent_path='F:\\University\\Final Project\\dataset\\new_dataset\\0',
                                model_name=processor.vgg_model,
                                fine_tune=True,
                                num_epochs=200,
                                fine_tune_epochs=20,
                                fine_tune_trainable_conv_layers=10,
                                input_shape=(32, 32),
                                train_batch_size=2048)
    #     processor.k_fold_cross_validation(x, y, 10, processor.resnet_model, (64, 64), 1200, 0, True, True)

    # processor.k_fold_cross_validation(x, y, k=10, model_name=processor.resnet_model,
    #                                   input_shape=(256, 256),
    #                                   output_dim=1,
    #                                   train_batch_size=1,
    #                                   augment_type=0,
    #                                   weighted_class=False,
    #                                   fine_tune=True,
    #                                   num_epochs=50,
    #                                   manual_augment_path=None)
    # processor.hold_out_method(x, y, model_name=processor.vgg_model,
    #                           input_shape=(32, 32),
    #                           output_dim=1,
    #                           train_batch_size=1024,
    #                           augment_type=0,
    #                           weighted_class=False,
    #                           fine_tune=True,
    #                           num_epochs=10,
    #                           manual_augment_path=None)

    # processor.k_fold_cross_validation(x, y, k=10, model_name=processor.CNN_model_7_layers,
    #                                   input_shape=(32, 32),
    #                                   train_batch_size=1024,
    #                                   augment_type=3,
    #                                   weighted_class=False,
    #                                   fine_tune=False,
    #                                   num_epochs=10,
    #                                   manual_augment_path='F:\\University\\Final Project\\dataset\\data-augmented\\1')

    # processor.train_manual_dataset(x, y, model_name=processor.CNN_model_7_layers,
    #                                input_shape=(32, 32),
    #                                train_batch_size=1024,
    #                                num_epochs=10,
    #                                fine_tune=False,
    #                                manual_path=None)


# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------show preprocessed slices----------------------------------------------
#     x = cv2.imread('F:\\University\\Final Project\\dataset\\data-augmented\\0\\0\\4.tiff')
#     cv2.imshow('x', x)
#     cv2.waitKey()
#     for patient in database.get_samples():
#         for sample in patient.get_examinations():
#             for slice_mri in sample.get_slices()[8:]:
#                 cv2.imshow("original", slice_mri.slice_image)
#                 cv2.waitKey(0)
#
#                 # no_skull = preprocessor.skull_stripping_1(slice_mri.slice_image)
#                 x, y, z = preprocessor.get_least_sized_image_encompassing_brain(slice_mri.slice_image, 16)
#                 cv2.imshow("preprocessed", x)
#                 cv2.waitKey(0)
# # ----------------------------------------------------------------------------------------------------------------------


# ------------------------------------------visualize slice with lesions------------------------------------------------
#     for patient in database.get_samples():
#         for examination in patient.get_examinations():
#             visualizer.show_brain_mri(examination)
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------show x's with label 1----------------------------------------------------
#     label_1, label_0 = [], []
#     for i in range(len(y)):
#         if y[i] == 1:
#             label_1.append(x[i])
#         else:
#             label_0.append(x[i])
#
#     for x in label_1:
#         cv2.imshow('1', x)
#         cv2.waitKey(0)
# ----------------------------------------------------------------------------------------------------------------------


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


if __name__ == '__main__':
    main()
