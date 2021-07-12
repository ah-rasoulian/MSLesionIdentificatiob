import preprocessor
from brainmri import BrainMRI
from brainmri import MRISlice
from brainmri import Patient
import scipy.io
import os
import cv2
import numpy


class Database:
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.samples = []

    def read_dataset(self):
        for item in sorted(os.listdir(self.parent_dir)):
            item_path = os.path.join(self.parent_dir, item)
            if os.path.isdir(item_path):
                self.add_new_sample(item_path)

    def get_samples(self):
        return self.samples

    def get_all_slices_with_labels(self):
        slices = []
        labels = []
        for patient in self.samples:
            patient: Patient
            for brian_mri in patient.get_examinations():
                brian_mri: BrainMRI
                for slice_mri in brian_mri.get_slices():
                    slice_mri: MRISlice
                    slices.append(slice_mri.get_slice_image())
                    labels.append(slice_mri.does_contain_lesion())

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
                    blank = numpy.zeros(slice_mri.get_slice_image().shape[0:2])
                    lesions_contour_marked_image = cv2.drawContours(blank.copy(), slice_mri.get_lesions(), -1, 1, -1)
                    for patch in preprocessor.get_image_patches(slice_mri.get_slice_image(), patch_width, patch_height,
                                                                horizontal_gap, vertical_gap):
                        patch_contour_points = [[patch.get_top_left_x(), patch.get_top_left_y()],
                                                [patch.get_top_left_x() + patch_width, patch.get_top_left_y()],
                                                [patch.get_top_left_x() + patch_width,
                                                 patch.get_top_left_y() + patch_height],
                                                [patch.get_top_left_x(), patch.get_top_left_y() + patch_height]]
                        patch_contour = numpy.array(patch_contour_points).reshape((-1, 1, 2)).astype(numpy.int32)
                        patch_contour_marked_image = cv2.drawContours(blank.copy(), [patch_contour], -1, 1, -1)

                        patch_lesion_intersection = numpy.logical_and(lesions_contour_marked_image,
                                                                      patch_contour_marked_image)

                        patches.append(patch.patch_image)
                        if patch_lesion_intersection.any():
                            labels.append(1)
                        else:
                            labels.append(0)
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

                # resize image into 512 * 512 if it is not so
                resized = False
                original_size = slice_image.shape
                if original_size != (512, 512):
                    resized = True
                    slice_image = cv2.resize(slice_image, (512, 512))

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
                            if resized:
                                _x = int(x[i][0] / original_size[1] * 512)
                                _y = int(y[i][0] / original_size[0] * 512)
                            else:
                                _x = x[i][0]
                                _y = y[i][0]
                            lesion_points.append([_x, _y])
                        lesion_contours = numpy.array(lesion_points).reshape((-1, 1, 2)).astype(numpy.int32)
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
