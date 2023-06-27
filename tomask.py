import os
import cv2

def invert_colors(folder_name):
    file_list = os.listdir(folder_name)
    for file_name in file_list:
        file_path = os.path.join(folder_name, file_name)
        image = cv2.imread(file_path)
        if image is not None:
            inverted_image = cv2.bitwise_not(image)
            cv2.imwrite(file_path, inverted_image)

            print(f"Inverted colors for {file_name}")
        else:
            print(f"Failed to read {file_name}")