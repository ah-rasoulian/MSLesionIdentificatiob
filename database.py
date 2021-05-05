from brainmri import Patient


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
            pass

        new_sample = Patient(brain_samples[0], brain_samples[1])
        self.samples.append(new_sample)

