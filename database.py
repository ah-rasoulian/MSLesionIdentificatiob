from brainmri import BrainMRI
from brainmri import MRISlice
from brainmri import Patient
import scipy.io
from os import listdir
import cv2
import numpy


class Database:
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.samples = []

    def read_dataset(self):
        pass

    def add_new_sample(self, sample_directory: str):
        examination_directories = [sample_directory + "/1/", sample_directory + "/2/"]
        brain_samples = []
        for directory in examination_directories:
            brain_examination = BrainMRI()

            # classifying all file names based on their extension, dividing them into slices and lesions files
            slices_files = []
            lesions_files = []
            for file_name in listdir(directory):
                if file_name.endswith(".TIF") or file_name.endswith(".bmp"):
                    slices_files.append(file_name)
                elif file_name.endswith(".plq"):
                    lesions_files.append(file_name)

            # creating new MRI slice and adding it into brain MRI
            for slice_file_name in slices_files:
                slice_file_path = directory + slice_file_name
                slice_image = cv2.imread(slice_file_path, cv2.IMREAD_GRAYSCALE)

                mri_slice = MRISlice(slice_image)
                slice_file_name_without_extension = slice_file_name.replace('.TIF', '')
                for lesion_file_name in lesions_files:
                    if lesion_file_name.startswith(slice_file_name_without_extension):
                        lesion_file_path = directory + lesion_file_name
                        lesion_file = scipy.io.loadmat(lesion_file_path)
                        lesion_points = []
                        x, y = lesion_file.get("xi"), lesion_file.get('yi')
                        for i in range(len(x)):
                            lesion_points.append([x[i][0], y[i][0]])
                        lesion_contours = numpy.array(lesion_points).reshape((-1, 1, 2)).astype(numpy.int32)
                        mri_slice.add_new_lesion(lesion_contours)

                # adding the slice into brain MRI
                brain_examination.add_new_slice(mri_slice)
            # adding brain examination into Patient object
            brain_samples.append(brain_examination)
        # adding created Patient object into dataset
        new_sample = Patient(brain_samples[0], brain_samples[1])
        self.samples.append(new_sample)
