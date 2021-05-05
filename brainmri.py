class Patient:
    def __init__(self, patient_code, first_examination, second_examination):
        self.patient_code = patient_code
        self.first_examination = first_examination
        self.second_examination = second_examination

    def get_examinations(self):
        return [self.first_examination, self.second_examination]


class BrainMRI:
    def __init__(self):
        self.slices = []

    def add_new_slice(self, new_slice):
        self.slices.append(new_slice)

    def get_slices(self):
        return self.slices


class MRISlice:
    def __init__(self, slice_image):
        self.slice_image = slice_image
        self.lesions = []

    def add_new_lesion(self, new_lesion):
        self.lesions.append(new_lesion)

    def get_slice_image(self):
        return self.slice_image

    def get_lesions(self):
        return self.lesions

    def does_contain_lesion(self):
        if len(self.lesions) > 0:
            return 1
        else:
            return 0
