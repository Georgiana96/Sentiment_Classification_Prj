import glob
import os
import cv2
import numpy as np

current_directory = os.path.dirname(__file__)

REMOVE_AXIS = False
DUPLICATE_SPEC = True
SOURCE_SPECTOGRAMS_DIRECTORY = "SPECTOGRAMS/original/"
DST_SPECTOGRAMS_DIRECTORY = "SPECTOGRAMS/removed_white_border/"
DST_SPECTOGRAMS_DIRECTORY2 = "SPECTOGRAMS/duplicated/"
TRAIN_SUBDIR = "train/"
TEST_SUBDIR = "test/"
VALIDATION_SUBDIR = "validation/"

ANGER_SUBDIR = "anger/"
NON_ANGER_SUBDIR = "non_anger/"

def remove_axis_from_images(source_directory_path, destination_directory_path):
    for image_path in glob.glob(source_directory_path + '*.png'):
        #image path is like C:/Faculta_an_4/Licenta/FriendsDB+Paper/MELD.Raw/train-model-project/SPECTOGRAMS/original/train/non_anger\dia404_utt0.png
        file_name = image_path.split('\\')[1]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cropped_img = img[58:429, 80:578].copy()
        cv2.imwrite(destination_directory_path + file_name, cropped_img)


# def resize_images(source_directory, destination_directory):
#     for relative_path in glob.glob(source_directory + '*.png'):
#         file_name = relative_path.split('\\')[1]
#         image_path = current_directory + '/' + source_directory + file_name
#         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         resized_img = cv2.resize(img, dsize=(NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
#         cv2.imwrite(current_directory + "/" + destination_directory + file_name, resized_img)
#

def get_the_first_white_pixel_coord(height, width, img_data):
    for x in range(height):
        for y in range(width):
            pixel = img_data[x, y]
            if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
                first_white_x = x
                first_white_y = y
                return [first_white_x, first_white_y]

    return [height, width]


## duplicate spectograms that don't have the axis
def duplicate_spec(source_directory_path, dst_directory_path):
    for image_path in glob.glob(source_directory_path + '*.png'):
        file_name = image_path.split('\\')[1]
        original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_data = np.asarray(original_img)  # img_data[row,column]
        img_height = img_data.shape[0]
        img_width = img_data.shape[1]
        [row, col] = get_the_first_white_pixel_coord(img_height, img_width, img_data)
        if [row, col] != [img_height, img_width]:
            overlaps = int(img_width / (col - 1)) - 1
            # img[y:y+h, x:x+w]
            cropped_img = original_img[0:0 + img_height, row:row + col - 1].copy()
            cropped_img_width = cropped_img.shape[1]
            # l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]]
            x_start = col - 1
            y_start = 0
            for i in range(overlaps):
                original_img[y_start:y_start + img_height, x_start:x_start + cropped_img_width] = cropped_img
                x_start = x_start + cropped_img_width - 2

            left_to_fill = img_width - x_start
            cropped_img2 = original_img[0:0 + img_height, row:row + left_to_fill].copy()
            original_img[y_start:y_start + img_height, x_start:x_start + left_to_fill] = cropped_img2

        cv2.imwrite(dst_directory_path + file_name, original_img)


########### REMOVE AXIS FROM IMAGES ############
if REMOVE_AXIS:
    src_spectograms_path = current_directory + "/" + SOURCE_SPECTOGRAMS_DIRECTORY
    dst_spectograms_path = current_directory + "/" + DST_SPECTOGRAMS_DIRECTORY

    print("GENERATING FOR TRAIN...")
    remove_axis_from_images(src_spectograms_path + TRAIN_SUBDIR + ANGER_SUBDIR,
                        dst_spectograms_path + TRAIN_SUBDIR + ANGER_SUBDIR)
    remove_axis_from_images(src_spectograms_path + TRAIN_SUBDIR + NON_ANGER_SUBDIR,
                        dst_spectograms_path + TRAIN_SUBDIR + NON_ANGER_SUBDIR)

    print("GENERATING FOR TEST...")
    remove_axis_from_images(src_spectograms_path + TEST_SUBDIR + ANGER_SUBDIR,
                        dst_spectograms_path + TEST_SUBDIR + ANGER_SUBDIR)
    remove_axis_from_images(src_spectograms_path + TEST_SUBDIR + NON_ANGER_SUBDIR,
                        dst_spectograms_path + TEST_SUBDIR + NON_ANGER_SUBDIR)

    print("GENERATING FOR VALIDATION...")
    remove_axis_from_images(src_spectograms_path + VALIDATION_SUBDIR + ANGER_SUBDIR,
                        dst_spectograms_path + VALIDATION_SUBDIR + ANGER_SUBDIR)
    remove_axis_from_images(src_spectograms_path + VALIDATION_SUBDIR + NON_ANGER_SUBDIR,
                        dst_spectograms_path + VALIDATION_SUBDIR + NON_ANGER_SUBDIR)



########## DUPLICATE SPECTOGRAMS THAT HAVE WHITE PADDING
if DUPLICATE_SPEC:
    src_spectograms_path = current_directory + "/" + DST_SPECTOGRAMS_DIRECTORY
    dst_spectograms_path = current_directory + "/" + DST_SPECTOGRAMS_DIRECTORY2

    print("GENERATING FOR TRAIN...")
    duplicate_spec(src_spectograms_path + TRAIN_SUBDIR + ANGER_SUBDIR,
                            dst_spectograms_path + TRAIN_SUBDIR + ANGER_SUBDIR)
    duplicate_spec(src_spectograms_path + TRAIN_SUBDIR + NON_ANGER_SUBDIR,
                            dst_spectograms_path + TRAIN_SUBDIR + NON_ANGER_SUBDIR)

    print("GENERATING FOR TEST...")
    duplicate_spec(src_spectograms_path + TEST_SUBDIR + ANGER_SUBDIR,
                            dst_spectograms_path + TEST_SUBDIR + ANGER_SUBDIR)
    duplicate_spec(src_spectograms_path + TEST_SUBDIR + NON_ANGER_SUBDIR,
                            dst_spectograms_path + TEST_SUBDIR + NON_ANGER_SUBDIR)

    print("GENERATING FOR VALIDATION...")
    duplicate_spec(src_spectograms_path + VALIDATION_SUBDIR + ANGER_SUBDIR,
                            dst_spectograms_path + VALIDATION_SUBDIR + ANGER_SUBDIR)
    duplicate_spec(src_spectograms_path + VALIDATION_SUBDIR + NON_ANGER_SUBDIR,
                            dst_spectograms_path + VALIDATION_SUBDIR + NON_ANGER_SUBDIR)



