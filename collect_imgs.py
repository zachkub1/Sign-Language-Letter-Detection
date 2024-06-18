import os
import cv2

DATA_DIR = 'C:/Users/ZacharyKublalsingh/Desktop/Sign Language/data_folder/data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36
dataset_size_per_class = 100 

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    count = 0
    while count < dataset_size_per_class:
        ret, frame = cap.read()

        if not ret:
            print("Error capturing frame. Exiting loop")
            break

        frame_filename = os.path.join(class_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {count} to {frame_filename}")
        count += 1

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
