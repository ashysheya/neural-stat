import cv2
import os
from tqdm import tqdm


def add_dir(new_dir):
    '''
    Used to create a new directory if it does not exist
    :param new_dir: directory to be created
    '''
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


print("processing")

dir_original = "C:/Users/Victor/Documents/Cambridge/Lent/MLMI4/Project/Poses/all_actions/"
dir_resized = "C:/Users/Victor/Documents/Cambridge/Lent/MLMI4/Project/Poses/all_actions_resized/"
add_dir(dir_resized)

resized_w = 64
resized_h = 64
dim_resized = (resized_w, resized_h)  # Resize images to 64x64

for video in tqdm(os.listdir(dir_original)):
    add_dir(dir_resized + video)

    video_dir = dir_original + video
    vidcap = cv2.VideoCapture(video_dir)
    success, image = vidcap.read()
    count = 0

    # Extract 60 frames
    while success and count<60:
        # Make square image 120x120
        cropped = image[:, 20:140]
        # Resize
        resized = cv2.resize(cropped, dim_resized, interpolation=cv2.INTER_AREA)
        cv2.imwrite(dir_resized + video + "/frame%d.jpg" % count, resized)  # save frame as JPEG file
        success, image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1
