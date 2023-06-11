import cv2
import os
import numpy as np

def detect_and_crop_handwriting(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    max_offset = -1
    max_offset_contour = None

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            cX = 0
            cY = 0
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        offset = np.sqrt((center_x - cX) ** 2 + (center_y - cY) ** 2)

        if offset > max_offset:
            max_offset = offset
            max_offset_contour = contour

    if max_offset_contour is not None:
        x, y, w, h = cv2.boundingRect(max_offset_contour)

        aspect_ratio = float(w) / h

        if aspect_ratio > 1:
            y_padding = int((w - h) / 2)
            x_padding = 0
        else:
            x_padding = int((h - w) / 2)
            y_padding = 0

        x -= x_padding
        w += 2 * x_padding
        y -= y_padding
        h += 2 * y_padding

        x = max(x, 0)
        w = min(w, width)
        y = max(y, 0)
        h = min(h, height)

        cropped_image = image[y:y + h, x:x + w]

        # resized_image = cv2.resize(cropped_image, (300, 300))

        # resized_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # resized_gray = resized_gray.reshape((28, 28, 1))

        return cropped_image

    else:
        print('No handwriting detected in the image.')
        return None

def convert_to_array(data_path,size):
    folders = ['0','1','2','3','4','5','6','7','8','9']
    X, y = [], []
    kernel = np.ones((5, 5), np.uint8)
    for folder in folders:
        folder_path = os.path.join(data_path, folder)
        images = os.listdir(folder_path)
        for image_name in images:
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.dilate(image, kernel, iterations=1)
            image = detect_and_crop_handwriting(image)
            image = cv2.resize(image, (size, size))  # Resize the image to 28x28 pixels
            X.append(image.flatten())  # Flatten the image and add it to the feature matrix
            y.append(int(folder))  # Add the corresponding label

    X_data = np.array(X)
    y_data = np.array(y)
    return X_data,y_data


# data = pickle.load(open("thainumber_{}.pkl".format(size), "rb"))
# X = data['X']
# Y = data['Y']