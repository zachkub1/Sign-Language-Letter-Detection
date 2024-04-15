import os

import cv2

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(1)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error capturing frame. Exiting loop")
            break

        class_index = dataset_size % number_of_classes
        frame_filename = os.path.join(DATA_DIR, str(class_index),f"frame_{dataset_size}.jpg")
        cv2.imwrite(frame_filename,frame)
        dataset_size -=1

cap.release()
cv2.destroyAllWindows()