from database import Database
import visualizer
import preprocessor
import processor

from collections import Counter
import cv2


def main():
    dataset_dir = "F://University/Final Project/dataset/Initial & repeat MRI in MS-Free Dataset"
    database = Database(dataset_dir)
    database.read_dataset()
    # database.add_new_sample("F://University/Final Project/dataset/Initial & repeat MRI in MS-Free Dataset/AA")
    #
    x, y = database.get_all_patches_with_labels(64, 64, 64, 64)
    print(len(x))
    processor.k_fold_cross_validation(x, y, 10, processor.deep_model_1)
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
