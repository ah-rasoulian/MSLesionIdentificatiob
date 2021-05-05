from database import Database
import visualizer
from collections import Counter


def main():
    dataset_dir = "/home/amirhossein/Data/University/Final Project/Work/dataset/Health Lab - University of Cyprus/Initial & repeat MRI in MS-Free Dataset"
    database = Database(dataset_dir)
    database.read_dataset()

    x, y = database.get_all_slices_with_labels()


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
