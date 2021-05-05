from database import Database


def main():
    dataset_dir = "/home/amirhossein/Data/University/Final Project/Work/dataset/Health Lab - University of Cyprus/Initial & repeat MRI in MS-Free Dataset"
    database = Database(dataset_dir)
    database.read_dataset()
    x = database.get_sample()
    print(len(x))


if __name__ == '__main__':
    main()
