import preprocessor
from brainmri import BrainMRI
from brainmri import MRISlice
from brainmri import Patient
import scipy.io
import os
import cv2
import numpy as np
import random


class Database:
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.samples = []
        self.x = []
        self.y = []

    def read_dataset(self):
        for item in sorted(os.listdir(self.parent_dir)):
            item_path = os.path.join(self.parent_dir, item)
            if os.path.isdir(item_path):
                self.add_new_sample(item_path)

    def read_images(self):
        x, y = [], []
        class_1_path = os.path.join(self.parent_dir, '1')
        class_0_path = os.path.join(self.parent_dir, '0')

        for new_image in os.listdir(class_1_path):
            new_image_path = os.path.join(class_1_path, new_image)
            x.append(cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE))
            y.append(1)

        for new_image in os.listdir(class_0_path):
            new_image_path = os.path.join(class_0_path, new_image)
            x.append(cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE))
            y.append(0)

        z = list(zip(x, y))
        random.shuffle(z)

        self.x, self.y = zip(*z)

    def get_images_with_labels(self):
        return np.array(self.x), np.array(self.y)

    def get_samples(self):
        return self.samples

    def get_all_slices_with_labels(self):
        slices = []
        labels = []
        total_lesions = 0
        for patient in self.samples:
            patient: Patient
            for brian_mri in patient.get_examinations():
                brian_mri: BrainMRI
                for slice_mri in brian_mri.get_slices():
                    slice_mri: MRISlice
                    slices.append(slice_mri.get_slice_image())
                    labels.append(slice_mri.does_contain_lesion())
                    total_lesions += len(slice_mri.get_lesions())

        print(total_lesions)
        return slices, labels

    def get_all_patches_with_labels(self, patch_width, patch_height, horizontal_gap=1, vertical_gap=1):
        patches = []
        labels = []

        for patient in self.samples:
            patient: Patient
            for brain_mri in patient.get_examinations():
                brain_mri: BrainMRI
                for slice_mri in brain_mri.get_slices():
                    slice_mri: MRISlice
                    blank = np.zeros(slice_mri.get_slice_image().shape[0:2])
                    lesions_contour_marked_image = cv2.drawContours(blank.copy(), slice_mri.get_lesions(), -1, 1, -1)
                    unique, counts = np.unique(lesions_contour_marked_image, return_counts=True)
                    total_number_of_lesions_pixels = dict(zip(unique, counts)).get(1, 0)
                    for patch in preprocessor.get_image_patches(slice_mri.get_slice_image(), patch_width, patch_height,
                                                                horizontal_gap, vertical_gap):
                        patch_contour_points = [[patch.get_top_left_x(), patch.get_top_left_y()],
                                                [patch.get_top_left_x() + patch_width, patch.get_top_left_y()],
                                                [patch.get_top_left_x() + patch_width,
                                                 patch.get_top_left_y() + patch_height],
                                                [patch.get_top_left_x(), patch.get_top_left_y() + patch_height]]
                        patch_contour = np.array(patch_contour_points).reshape((-1, 1, 2)).astype(np.int32)
                        patch_contour_marked_image = cv2.drawContours(blank.copy(), [patch_contour], -1, 1, -1)

                        patch_lesion_intersection = np.logical_and(lesions_contour_marked_image,
                                                                   patch_contour_marked_image)
                        unique, counts = np.unique(patch_lesion_intersection, return_counts=True)
                        patch_number_of_lesions_pixels = dict(zip(unique, counts)).get(1, 0)

                        patches.append(patch.patch_image)
                        if patch_number_of_lesions_pixels > total_number_of_lesions_pixels / 2 or patch_number_of_lesions_pixels >= (
                                0.05 * patch_width * patch_height):
                            labels.append(1)
                        else:
                            labels.append(0)
        return patches, labels

    def get_lesions_sizes(self):
        sizes = []
        for patient in self.samples:
            patient: Patient
            for brain_mri in patient.get_examinations():
                brain_mri: BrainMRI
                for slice_mri in brain_mri.get_slices():
                    slice_mri: MRISlice
                    if slice_mri.does_contain_lesion():
                        blank = np.zeros(slice_mri.get_slice_image().shape[0:2])
                        for lesion_id in range(len(slice_mri.get_lesions())):
                            new_lesion_contour_marked_image = cv2.drawContours(blank.copy(), slice_mri.get_lesions(),
                                                                               lesion_id, 1, -1)
                            unique, counts = np.unique(new_lesion_contour_marked_image, return_counts=True)

                            sizes.append(dict(zip(unique, counts)).get(1, 0))

        return sizes

    def get_lesion_rectangle_dimensions(self):
        heights, widths, areas = [], [], []
        for patient in self.samples:
            patient: Patient
            for brain_mri in patient.get_examinations():
                brain_mri: BrainMRI
                for slice_mri in brain_mri.get_slices():
                    for lesion in (slice_mri.get_lesions()):
                        x, y, w, h = cv2.boundingRect(lesion)
                        widths.append(w)
                        heights.append(h)
                        areas.append(h * w)

        return heights, widths, areas

    def get_patches_of_affected_slices_with_labels(self, patch_width, patch_height, horizontal_gap=1, vertical_gap=1):
        patches = []
        labels = []
        number_of_slices_processed = 0
        for patient in self.samples:
            patient: Patient
            for brain_mri in patient.get_examinations():
                brain_mri: BrainMRI
                for slice_mri in brain_mri.get_slices():
                    slice_mri: MRISlice
                    if slice_mri.does_contain_lesion():
                        blank = np.zeros(slice_mri.get_slice_image().shape[0:2])
                        lesions_contour_marked_image = []
                        total_number_of_lesions_pixels = []
                        for lesion_id in range(len(slice_mri.get_lesions())):
                            new_lesion_contour_marked_image = cv2.drawContours(blank.copy(), slice_mri.get_lesions(),
                                                                               lesion_id, 1, -1)
                            unique, counts = np.unique(new_lesion_contour_marked_image, return_counts=True)

                            lesions_contour_marked_image.append(new_lesion_contour_marked_image)
                            total_number_of_lesions_pixels.append(dict(zip(unique, counts)).get(1, 0))

                        for patch in preprocessor.get_image_patches(slice_mri.get_slice_image(), patch_width,
                                                                    patch_height,
                                                                    horizontal_gap, vertical_gap):
                            patch_contour_points = [[patch.get_top_left_x(), patch.get_top_left_y()],
                                                    [patch.get_top_left_x() + patch_width, patch.get_top_left_y()],
                                                    [patch.get_top_left_x() + patch_width,
                                                     patch.get_top_left_y() + patch_height],
                                                    [patch.get_top_left_x(), patch.get_top_left_y() + patch_height]]
                            patch_contour = np.array(patch_contour_points).reshape((-1, 1, 2)).astype(np.int32)
                            patch_contour_marked_image = cv2.drawContours(blank.copy(), [patch_contour], -1, 1, -1)

                            total_patch_number_of_lesions = 0
                            for lesion_contour_marked_id in range(len(lesions_contour_marked_image)):
                                patch_lesion_intersection = np.logical_and(
                                    lesions_contour_marked_image[lesion_contour_marked_id],
                                    patch_contour_marked_image)

                                unique, counts = np.unique(patch_lesion_intersection, return_counts=True)
                                patch_number_of_lesions_pixels = dict(zip(unique, counts)).get(1, 0)
                                total_patch_number_of_lesions += patch_number_of_lesions_pixels

                                if patch_number_of_lesions_pixels == total_number_of_lesions_pixels[
                                    lesion_contour_marked_id]:
                                    patches.append(patch.patch_image)
                                    labels.append(1)
                                    break
                            if total_patch_number_of_lesions == 0:
                                patches.append(patch.patch_image)
                                labels.append(0)

                        number_of_slices_processed += 1
                        print(number_of_slices_processed)

        return patches, labels

    def add_new_sample(self, sample_directory: str):
        examination_directories = [sample_directory + "/1/", sample_directory + "/2/"]
        brain_samples = []
        for directory in examination_directories:
            brain_examination = BrainMRI()

            # classifying all file names based on their extension, dividing them into slices and lesions files
            slices_files = []
            lesions_files = []
            for file_name in sorted(os.listdir(directory)):
                if file_name.endswith(".TIF") or file_name.endswith(".bmp"):
                    slices_files.append(file_name)
                elif file_name.endswith(".plq"):
                    lesions_files.append(file_name)

            # creating new MRI slice and adding it into brain MRI
            for slice_file_name in slices_files:
                slice_file_path = os.path.join(directory, slice_file_name)
                slice_image = cv2.imread(slice_file_path, cv2.IMREAD_GRAYSCALE)

                # apply preprocessing to slice image
                slice_image, transformed_x, transformed_y = preprocessor.pre_processing(slice_image)

                mri_slice = MRISlice(slice_image)
                if slice_file_name.endswith('.TIF'):
                    slice_file_name_without_extension = slice_file_name.replace('.TIF', '')
                else:
                    slice_file_name_without_extension = slice_file_name.replace('.bmp', '')

                for lesion_file_name in lesions_files:
                    if lesion_file_name.startswith(slice_file_name_without_extension):
                        lesion_file_path = os.path.join(directory, lesion_file_name)
                        lesion_file = scipy.io.loadmat(lesion_file_path)
                        lesion_points = []
                        x, y = lesion_file.get("xi"), lesion_file.get('yi')
                        for i in range(len(x)):
                            _x = x[i][0] - transformed_x
                            _y = y[i][0] - transformed_y
                            lesion_points.append([_x, _y])
                        lesion_contours = np.array(lesion_points).reshape((-1, 1, 2)).astype(np.int32)
                        mri_slice.add_new_lesion(lesion_contours)

                # adding the slice into brain MRI
                brain_examination.add_new_slice(mri_slice)
            # adding brain examination into Patient object
            brain_samples.append(brain_examination)
        # adding created Patient object into dataset
        sample_directory_parts = sample_directory.split('/')
        patient_code = sample_directory_parts[len(sample_directory_parts) - 1]
        new_sample = Patient(patient_code, brain_samples[0], brain_samples[1])

        self.samples.append(new_sample)
