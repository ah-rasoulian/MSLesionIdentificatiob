import cv2

import preprocessor
from brainmri import BrainMRI
from brainmri import MRISlice
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

intab = '1234567890'
outtab = '۱۲۳۴۵۶۷۸۹۰'
translation_table = str.maketrans(intab, outtab)


def show_single_slice(slice_mri: MRISlice, slice_number, total_slices_number, with_lesions=False):
    result_image = cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR)
    lesions = slice_mri.get_lesions()
    height, width, channel = result_image.shape

    # typing number of lesions
    cv2.putText(result_image, len(lesions).__str__(), (width // 40, 5 * height // 10), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.8, (255, 0, 255), 1)

    # typing the slice number over total slices
    cv2.putText(result_image, slice_number.__str__() + " / " + total_slices_number.__str__(),
                (width // 40, 6 * height // 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 255), 1)

    # drawing lesions contours if it is needed
    if with_lesions:
        cv2.drawContours(result_image, slice_mri.get_lesions(), -1, (0, 0, 255), 1)

    cv2.imshow("MRI", result_image)

    key = cv2.waitKey(0)  # wait for a keystroke in the window

    while key not in [27, 97, 100, 119, 115]:
        key = cv2.waitKey(0)

    if key == 119:
        return show_single_slice(slice_mri, slice_number, total_slices_number, True)
    elif key == 115:
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
        elif next_instruction == 97 and slice_number > 0:
            slice_number -= 1
        elif next_instruction == 100 and slice_number < len(slices) - 1:
            slice_number += 1


def show_sample_total_brain_mri(brain_mri: BrainMRI):
    rows, cols = 5, 5
    plt.figure(figsize=(10, 12))
    gs1 = gridspec.GridSpec(rows, cols)

    for i, slice_mri in enumerate(brain_mri.get_slices()):
        if i == rows * cols:
            break
        ax1 = plt.subplot(gs1[i])
        plt.axis('off')
        ax1.set_aspect('equal')
        ax1.set_title((i + 1).__str__().translate(translation_table), color='black', y=-0.2, x=0.5, fontsize=14)
        ax1.imshow(slice_mri.get_slice_image(), cmap='gray')

    plt.subplots_adjust(left=0.1, bottom=0.1, top=1, right=1, wspace=0.01, hspace=0.2)
    plt.show()


def show_slices_with_and_without_lesion(brain_mri: BrainMRI):
    plt.figure(figsize=(8, 8))
    gs1 = gridspec.GridSpec(2, 2)

    no_les = plt.subplot(gs1[1])
    no_les.set_aspect('equal')
    no_les.set_title('آ', color='black', y=-0.1, x=0.5, fontsize=14)
    no_les.axis('off')
    no_les_shown = False
    one_les = plt.subplot(gs1[0])
    one_les.set_aspect('equal')
    one_les.set_title('ب', color='black', y=-0.1, x=0.5, fontsize=14)
    one_les.axis('off')
    one_les_shown = False
    two_les = plt.subplot(gs1[3])
    two_les.set_aspect('equal')
    two_les.set_title('پ', color='black', y=-0.1, x=0.5, fontsize=14)
    two_les.axis('off')
    two_les_shown = False
    three_les = plt.subplot(gs1[2])
    three_les.set_aspect('equal')
    three_les.set_title('ت', color='black', y=-0.1, x=0.5, fontsize=14)
    three_les.axis('off')
    three_les_shown = False
    for i, slice_mri in enumerate(brain_mri.get_slices()[6:]):
        if len(slice_mri.get_lesions()) == 0 and no_les_shown is False:
            s = cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR)
            s = cv2.drawContours(s, slice_mri.get_lesions(), -1, (255, 0, 0), 1)
            no_les.imshow(s)
            no_les_shown = True
        if len(slice_mri.get_lesions()) == 1 and one_les_shown is False:
            s = cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR)
            s = cv2.drawContours(s, slice_mri.get_lesions(), -1, (255, 0, 0), 1)
            one_les.imshow(s)
            one_les_shown = True
        if len(slice_mri.get_lesions()) == 2 and two_les_shown is False:
            s = cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR)
            s = cv2.drawContours(s, slice_mri.get_lesions(), -1, (255, 0, 0), 1)
            two_les.imshow(s)
            two_les_shown = True
        if len(slice_mri.get_lesions()) == 3 and three_les_shown is False:
            s = cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR)
            s = cv2.drawContours(s, slice_mri.get_lesions(), -1, (255, 0, 0), 1)
            three_les.imshow(s)
            three_les_shown = True
        if no_les_shown is True and one_les_shown is True and two_les_shown is True and three_les_shown is True:
            break

    plt.subplots_adjust(left=0.1, bottom=0.1, top=1, right=1, wspace=0.01, hspace=0.15)
    plt.show()


def clarify_slice_contour(slice_mri: MRISlice):
    fig = plt.figure(figsize=(8, 10))
    gs1 = gridspec.GridSpec(2, 2, width_ratios=[1, 4])

    lesion_marked_image = cv2.drawContours(cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR),
                                           slice_mri.get_lesions(),
                                           -1, (255, 0, 0), 1)
    x, y, w, h = cv2.boundingRect(slice_mri.get_lesions()[0])
    offset = 16
    lesion_marked_image = cv2.rectangle(lesion_marked_image, (x - offset, y - offset), (x + w + offset, y + h + offset),
                                        (0, 200, 0), 1)
    ax1 = fig.add_subplot(gs1[0, 1])
    ax1.imshow(lesion_marked_image)
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_title('آ', color='black', y=-0.1, x=0.5, fontsize=14)

    lesion_rect = cv2.cvtColor(slice_mri.get_slice_image()[y - offset: y + h + offset, x - offset: x + w + offset],
                               cv2.COLOR_GRAY2BGR)
    original_size = lesion_rect.shape
    resized_size = (512, 512)
    lesion_rect = cv2.resize(lesion_rect, resized_size)

    ax3 = fig.add_subplot(gs1[:, 0])
    text_kwargs = dict(ha='center', va='center', fontsize=8, color='black')

    # ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('پ', color='black', y=-0.05, x=0.5, fontsize=14)

    for i, point in enumerate(slice_mri.get_lesions()[0]):
        center = (int((point[0][0] - x + offset) / original_size[1] * resized_size[1]),
                  int((point[0][1] - y + offset) / original_size[0] * resized_size[0]))
        lesion_rect = cv2.circle(lesion_rect, center, 3, (255, 0, 0), -1)
        lesion_rect = cv2.putText(lesion_rect, str(point[0]), center, cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))

        ax3.text(0.5, 0.85 - i / 17, str(point[0]), **text_kwargs)

    ax2 = fig.add_subplot(gs1[1, 1])
    ax2.imshow(lesion_rect)
    ax2.axis('off')
    ax2.set_aspect('equal')
    ax2.set_title('ب', color='black', y=-0.1, x=0.5, fontsize=14)

    plt.subplots_adjust(left=0, bottom=0.075, top=1, right=0.7, wspace=0.01, hspace=0.1)
    plt.show()


def show_data_augmented(slice_mri: MRISlice):
    image = slice_mri.get_slice_image()
    # rotation
    plt.figure(figsize=(10, 3))
    gs1 = gridspec.GridSpec(3, 10)

    for i in range(30):
        angle = (15 - i) * np.pi / 180
        ax1 = plt.subplot(gs1[i])
        plt.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(preprocessor.image_rotation(image, angle), cmap='gray')
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)
    print('rotation')
    plt.show()

    plt.figure(figsize=(10, 3))
    gs2 = gridspec.GridSpec(3, 10)
    for i, gamma in enumerate(range(70, 130, 2)):
        gamma = gamma / 100
        ax1 = plt.subplot(gs2[i])
        plt.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(preprocessor.image_gamma_correction(image, gamma), cmap='gray')
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)
    print('gamma')
    plt.show()

    plt.figure(figsize=(10, 3))
    gs3 = gridspec.GridSpec(3, 10)
    for i in range(30):
        ax1 = plt.subplot(gs3[i])
        plt.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(preprocessor.image_gaussian_noise_injection(image, 0, 0.01) * 255, cmap='gray')
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)
    print('noise')
    plt.show()

    plt.figure(figsize=(10, 3))
    gs4 = gridspec.GridSpec(3, 10)
    for i, j in enumerate(np.random.randint(-100, 100, 30)):
        if j < 0:
            width_shift, height_shift = -(-j // 10), -(-j % 10)
        else:
            width_shift, height_shift = j // 10, j % 10
        ax1 = plt.subplot(gs4[i])
        plt.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(preprocessor.image_translation(image, width_shift, height_shift), cmap='gray')
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)
    print('translate')
    plt.show()

    plt.figure(figsize=(10, 3))
    gs5 = gridspec.GridSpec(3, 10)
    for i, scale_factor in enumerate(range(70, 130, 2)):
        scale_factor = scale_factor / 100
        ax1 = plt.subplot(gs5[i])
        plt.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(preprocessor.clipped_zoom(image, scale_factor), cmap='gray')
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)
    print('zoom')
    plt.show()


def draw_lesion_rect_histogram(widths, heights):
    fig, ax = plt.subplots(figsize=(10, 7))

    x_bins = np.linspace(min(widths), max(widths), 50)
    y_bins = np.linspace(min(heights), max(heights), 50)
    plt.hist2d(widths, heights, bins=[x_bins, y_bins])
    plt.colorbar()

    ax.set_xlabel('width')
    ax.set_ylabel('height')
    plt.show()
