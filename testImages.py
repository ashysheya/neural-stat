import os
import cv2
import numpy as np
from tqdm import tqdm

dir_resized = "C:/Users/Victor/Documents/Cambridge/Lent/MLMI4/Project/YouTubeFaces/resized_images_DB/"

for person in os.listdir(dir_resized):
    for video in os.listdir(dir_resized+person):
        for frame in os.listdir(dir_resized + person + "/" + video):
            img_dir = dir_resized + person + "/" + video + "/" + frame
            img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED).astype(np.float64)/255
            img_s = img*255
            cv2.imshow('ImageWindow', img_s.astype(np.uint8))
            cv2.waitKey()
            testarray = np.array([img.transpose(2, 0, 1), img.transpose(2, 0, 1)])
            print(testarray)
            break
        break
    break
