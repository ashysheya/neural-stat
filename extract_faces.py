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


dir_original = "C:/Users/Victor/Documents/Cambridge/Lent/MLMI4/Project/extended-cohn-kanade-images/cohn-kanade-images/"
dir_resized = "C:/Users/Victor/Documents/Cambridge/Lent/MLMI4/Project/extended-cohn-kanade-images/resized/"

face_classifier_path = "C:/Users/Victor/Documents/Cambridge/Lent/MLMI4/Project/haarcascade_frontalface_default.xml"
# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(face_classifier_path)

# Specify dimensions of resized image
resized_w = 64
resized_h = 64
dim_resized = (resized_w, resized_h)

add_dir(dir_resized)
for person in tqdm(os.listdir(dir_original)):
    add_dir(dir_resized + person)
    for video in os.listdir(dir_original + person):
        if video == ".DS_Store":
            continue
        add_dir(dir_resized + person + "/" + video)

        for frame in os.listdir(dir_original + person + "/" + video)[-7:]:  # Take last 7 images
            img_dir = dir_original + person + "/" + video + "/" + frame
            save_dir = dir_resized + person + "/" + video + "/" + frame

            # Read the input image
            img = cv2.imread(img_dir)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert into grayscale

            # Detect and extract faces
            faces = face_cascade.detectMultiScale(gray, minNeighbors=3)
            if len(faces) == 0:
                print("Could not find face in frame ", frame)
                continue

            img_face = gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]

            # Check if not square, or if not enough images in dataset
            if img_face.shape[0] != img_face.shape[1]:
                raise Exception("Image is not square")

            # Resize image
            resized = cv2.resize(img_face, dim_resized, interpolation=cv2.INTER_AREA)
            cv2.imwrite(save_dir, resized)  # Save cropped and resized image
