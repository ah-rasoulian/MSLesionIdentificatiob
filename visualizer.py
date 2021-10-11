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
    # cv2.putText(result_image, slice_number.__str__() + " / " + total_slices_number.__str__(),
    #             (width // 40, 6 * height // 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 255), 1)

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


def show_patching_example(slice_mri: MRISlice):
    width, height = 36, 36
    width_gap, height_gap = 4, 4
    fig = plt.figure()
    gs1 = gridspec.GridSpec(2, 4)

    lesion_marked_image = cv2.drawContours(cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR),
                                           slice_mri.get_lesions(),
                                           -1, (255, 0, 0), -1)

    x, y, w, h = cv2.boundingRect(slice_mri.get_lesions()[0])

    x, y = x + w // 2 - width // 2, y + h // 2 - height // 2

    original_image = cv2.resize(lesion_marked_image[y: y + height, x: x + width], (10 * width, 10 * height))

    original_image = cv2.rectangle(original_image, (0, 0), (320, 320), (0, 255, 0), 1)
    original_image = cv2.rectangle(original_image, (0, 40), (320, 360), (0, 255, 0), 1)
    original_image = cv2.rectangle(original_image, (40, 0), (360, 320), (0, 255, 0), 1)
    original_image = cv2.rectangle(original_image, (40, 40), (360, 360), (0, 255, 0), 1)

    ax1 = fig.add_subplot(gs1[0, 1:3])
    ax1.imshow(original_image)
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_title('آ', color='black', y=-0.15, x=0.5, fontsize=14)

    image_number = 0
    labels = ['ب', 'پ', 'ت', 'ث']
    for j in range(y, y + height - 32 + 1, height_gap):
        for i in range(x, x + width - 32 + 1, width_gap):
            ax = fig.add_subplot(gs1[1, image_number])
            image = cv2.resize(lesion_marked_image[j: j + 32, i: i + 32], (10 * 32, 10 * 32))
            ax.imshow(image)
            ax.axis('off')
            ax.set_aspect('equal')
            ax.set_title(labels[3 - image_number], color='black', y=-0.25, x=0.5, fontsize=14)
            image_number += 1

    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0.1, hspace=0)
    plt.show()


def show_lesion_between_patches(slice_mri: MRISlice):
    width, height = 64, 32
    width_gap, height_gap = 32, 32
    fig = plt.figure()
    gs1 = gridspec.GridSpec(2, 4)

    lesion_marked_image = cv2.drawContours(cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR),
                                           slice_mri.get_lesions(),
                                           -1, (255, 0, 0), -1)

    x, y, w, h = cv2.boundingRect(slice_mri.get_lesions()[0])

    x, y = x + w // 2 - width // 2, y + h // 2 - height // 2

    original_image = cv2.resize(lesion_marked_image[y: y + height, x: x + width], (10 * width, 10 * height))

    original_image = cv2.rectangle(original_image, (0, 0), (320, 320), (0, 255, 0), 5)
    original_image = cv2.rectangle(original_image, (320, 0), (640, 320), (0, 255, 0), 5)

    ax1 = fig.add_subplot(gs1[0, 1:3])
    ax1.imshow(original_image)
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_title('آ', color='black', y=-0.25, x=0.5, fontsize=14)

    image_number = 0
    labels = ['پ', 'ب', 'ت', 'ث']
    for j in range(y, y + height - 32 + 1, height_gap):
        for i in range(x, x + width - 32 + 1, width_gap):
            ax = fig.add_subplot(gs1[1, 1 + image_number])
            image = cv2.resize(lesion_marked_image[j: j + 32, i: i + 32], (10 * 32, 10 * 32))
            ax.imshow(image)
            ax.axis('off')
            ax.set_aspect('equal')
            ax.set_title(labels[image_number], color='black', y=-0.25, x=0.5, fontsize=14)
            image_number += 1

    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0.1, hspace=0)
    plt.show()


def show_labeling_example(slice_mri: MRISlice):
    lesion_marked_image = cv2.drawContours(cv2.cvtColor(slice_mri.get_slice_image(), cv2.COLOR_GRAY2BGR),
                                           slice_mri.get_lesions(),
                                           -1, (255, 0, 0), 2)

    x, y, w, h = cv2.boundingRect(slice_mri.get_lesions()[0])
    patch_contour_1 = [[x, y], [x + 32, y], [x + 32, y + 32], [x, y + 32]]
    patch_contour_1 = np.array(patch_contour_1).reshape((-1, 1, 2)).astype(np.int32)
    patch_marked_image = cv2.drawContours(lesion_marked_image.copy(), [patch_contour_1], -1, (0, 255, 0), 2)

    patch_contour_2 = [[x + 96, y - 96], [x + 128, y - 96], [x + 128, y - 64], [x + 96, y - 64]]
    patch_contour_2 = np.array(patch_contour_2).reshape((-1, 1, 2)).astype(np.int32)
    patch_marked_image2 = cv2.drawContours(lesion_marked_image.copy(), [patch_contour_2], -1, (0, 255, 0), 2)
    blank = np.zeros(slice_mri.get_slice_image().shape[0:2])

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, hspace=0.2)

    gs1 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs[1])

    ax_in1_1 = plt.subplot(gs1[1:3, 0])
    ax_in1_1.imshow(patch_marked_image)
    ax_in1_1.axis('off')
    ax_in1_1.set_aspect('equal')

    ax_in1_2 = plt.subplot(gs1[0:2, 1])
    lesion_image = cv2.drawContours(blank.copy(), slice_mri.get_lesions(), -1, 1, -1)
    ax_in1_2.imshow(lesion_image, cmap='gray')
    ax_in1_2.axis('off')
    ax_in1_2.set_aspect('equal')

    ax_in1_3 = plt.subplot(gs1[2:4, 1])
    patch_image = cv2.drawContours(blank.copy(), [patch_contour_1], -1, 1, -1)
    ax_in1_3.imshow(patch_image, cmap='gray')
    ax_in1_3.axis('off')
    ax_in1_3.set_aspect('equal')

    ax_in1_4 = plt.subplot(gs1[1:3, 2])
    patch_lesion_intersection = np.logical_and(lesion_image, patch_image)
    ax_in1_4.imshow(patch_lesion_intersection, cmap='gray')
    ax_in1_4.axis('off')
    ax_in1_4.set_aspect('equal')

    ax_in2_1 = plt.subplot(gs2[1:3, 0])
    ax_in2_1.imshow(patch_marked_image2)
    ax_in2_1.axis('off')
    ax_in2_1.set_aspect('equal')

    ax_in2_2 = plt.subplot(gs2[0:2, 1])
    lesion_image = cv2.drawContours(blank.copy(), slice_mri.get_lesions(), -1, 1, -1)
    ax_in2_2.imshow(lesion_image, cmap='gray')
    ax_in2_2.axis('off')
    ax_in2_2.set_aspect('equal')

    ax_in2_3 = plt.subplot(gs2[2:4, 1])
    patch_image = cv2.drawContours(blank.copy(), [patch_contour_2], -1, 1, -1)
    ax_in2_3.imshow(patch_image, cmap='gray')
    ax_in2_3.axis('off')
    ax_in2_3.set_aspect('equal')

    ax_in2_4 = plt.subplot(gs2[1:3, 2])
    patch_lesion_intersection = np.logical_and(lesion_image, patch_image)
    ax_in2_4.imshow(patch_lesion_intersection, cmap='gray')
    ax_in2_4.axis('off')
    ax_in2_4.set_aspect('equal')

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


def show_least_sized_rectangle(slice_mri: MRISlice):
    plt.figure(figsize=(4, 8))
    least_size_image = preprocessor.get_least_sized_image_encompassing_brain(slice_mri.get_slice_image(), 0)[0]
    gs1 = gridspec.GridSpec(1, 2, width_ratios=[least_size_image.shape[1], slice_mri.get_slice_image().shape[1]])

    ax1 = plt.subplot(gs1[1])
    plt.axis('off')
    ax1.set_aspect('equal')
    ax1.set_title('آ', color='black', y=-0.2, x=0.5, fontsize=14)
    ax1.imshow(slice_mri.get_slice_image(), cmap='gray')

    ax2 = plt.subplot(gs1[0])
    plt.axis('off')
    ax2.set_aspect('equal')
    ax2.set_title('ب', color='black', y=-0.55, x=0.5, fontsize=14)
    ax2.imshow(least_size_image, cmap='gray')

    # plt.subplots_adjust(left=0.1, bottom=0.1, top=1, right=1, wspace=0.01, hspace=0.2)
    plt.show()


def make_background_white(image):
    white_bg = image.copy()
    white_bg[white_bg < 20] = 255

    plt.imshow(white_bg, cmap='gray')
    plt.show()
