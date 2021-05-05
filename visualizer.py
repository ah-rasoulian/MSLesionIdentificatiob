import cv2
from brainmri import BrainMRI
from brainmri import MRISlice


def show_single_slice(slice_mri: MRISlice, slice_number, total_slices_number, with_lesions=False):
    result_image = cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR)
    lesions = slice_mri.get_lesions()
    height, width, channel = result_image.shape

    # typing the status of the slice
    cv2.putText(result_image, slice_mri.does_contain_lesion().__str__(), (width//40, 4*height//10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 255), 1)

    # typing number of lesions
    cv2.putText(result_image, len(lesions).__str__(), (width//40, 5*height//10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 255), 1)

    # typing the slice number over total slices
    cv2.putText(result_image, slice_number.__str__() + " / " + total_slices_number.__str__(), (width//40, 6*height//10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 255), 1)

    # drawing lesions contours if it is needed
    if with_lesions:
        cv2.drawContours(result_image, slice_mri.get_lesions(), -1, (0, 0, 255), 1)

    cv2.imshow("MRI", result_image)

    key = cv2.waitKey(0)  # wait for a keystroke in the window
    while key not in [27, 81, 82, 83, 84]:
        key = cv2.waitKey(0)

    if key == 82:
        return show_single_slice(slice_mri, slice_number, total_slices_number, True)
    elif key == 84:
        return show_single_slice(slice_mri, slice_number, total_slices_number, False)
    else:
        return key


def show_brain_mri(brain_mri: BrainMRI):
    slice_number = 0
    slices = brain_mri.get_slices()

    while True:
        next_instruction = show_single_slice(slices[slice_number], slice_number, len(slices) - 1)
        if next_instruction == 27:
            break
        elif next_instruction == 81 and slice_number > 0:
            slice_number -= 1
        elif next_instruction == 83 and slice_number < len(slices) - 1:
            slice_number += 1
