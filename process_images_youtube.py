import os
import cv2
from tqdm import tqdm


def add_dir(new_dir):
    '''
    Used to create a new directory if it does not exist
    :param new_dir: directory to be created
    '''
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


dir_original = "C:/Users/Victor/Documents/Cambridge/Lent/MLMI4/Project/YouTubeFaces/aligned_images_DB/"
dir_resized = "C:/Users/Victor/Documents/Cambridge/Lent/MLMI4/Project/YouTubeFaces/resized_images_DB/"
add_dir(dir_resized)

zoom_factor = 2.2  # Crop to face - original image is 2.2 times the bounding box
resized_w = 64
resized_h = 64
dim_resized = (resized_w, resized_h)  # Resize images to 64x64

for person in tqdm(os.listdir(dir_original)):
    add_dir(dir_resized + person)
    for video in os.listdir(dir_original+person):
        add_dir(dir_resized + person + "/" + video)
        for frame in os.listdir(dir_original + person + "/" + video):
            img_dir = dir_original + person + "/" + video + "/" + frame
            save_dir = dir_resized + person + "/" + video + "/" + frame

            img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
            img_dim = img.shape
            cropped = img[round(img_dim[0] * (1 - 1 / zoom_factor) / 2):round(img_dim[0] * (1 + 1 / zoom_factor) / 2),
                      round(img_dim[1] * (1 - 1 / zoom_factor) / 2):round(img_dim[1] * (1 + 1 / zoom_factor) / 2)]
            resized = cv2.resize(cropped, dim_resized, interpolation=cv2.INTER_AREA)
            cv2.imwrite(save_dir, resized)  # Save cropped and resized image
