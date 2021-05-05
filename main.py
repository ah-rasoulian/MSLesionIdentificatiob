from database import Database
import visualizer


def main():
    dataset_dir = "/home/amirhossein/Data/University/Final Project/Work/dataset/Health Lab - University of Cyprus/Initial & repeat MRI in MS-Free Dataset"
    database = Database(dataset_dir)
    database.read_dataset()
    x = database.get_samples()
    visualizer.show_brain_mri(x[0].get_examinations()[0])


if __name__ == '__main__':
    main()
