class Patient:
    def __init__(self, first_examination, second_examination):
        self.first_examination = first_examination
        self.second_examination = second_examination


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

    def contains_lesion(self):
        if len(self.lesions) > 0:
            return True
        else:
            return False
