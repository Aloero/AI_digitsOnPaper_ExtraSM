import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class Detection:
    def __init__(self, path):
        self.model = load_model('path/to/model')

        self.path = path
        self.image = cv2.imread(self.path)

        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([70,10,30])
        upper_blue = np.array([170,255,150])  

        mask_blue = cv2.inRange(self.hsv, lower_blue, upper_blue)
        self.edges = mask_blue

    def show_images(self):
        self.reducing_picture()
        cv2.imshow('Original Image', self.image)
        cv2.imshow('Edge1 Image', self.edges)
        cv2.waitKey(0)
    

    def reducing_picture(self):
        self.image = cv2.resize(self.image, (1080//2, 1920//2))
        self.edges = cv2.resize(self.edges, (1080//2, 1920//2))

    def detections(self):
        kernel = np.ones((4,3),np.uint8)
        dilated = cv2.dilate(self.edges, kernel, iterations = 2)

        self.contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lower_white = np.array([120, 120, 120])
        upper_white = np.array([255, 255, 255])

        mask = cv2.inRange(self.image, lower_white, upper_white)

        lower_white2 = np.array([0, 0, 0])
        upper_white2 = np.array([180, 120, 100])

        mask2 = cv2.inRange(self.image, lower_white2, upper_white2)
        threshold_percent = 0.005 
      
        countours = []
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)

            total_pixels = w * h

            mask_pixels = np.sum(mask[y:y+h, x:x+w] > 0)
            mask2_pixels = np.sum(mask2[y:y+h, x:x+w] > 0)

            mask_percent = mask_pixels / total_pixels
            mask2_percent = mask2_pixels / total_pixels

            if mask_percent > threshold_percent and mask2_percent > threshold_percent:
                k = w / h
                if k > 0.3 and k < 1.3 and 25 < w < 400 and 25 < h < 400:

                    countours.append((x, y, w, h))

        threshold = 50

        def sort_contours(contour):
            x, y, _, _ = contour
            return (round(y / threshold), x)

        sorted_contours = sorted(countours, key=sort_contours)

        for countour in sorted_contours:
            x, y, w, h = countour
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 3)

            cropped_img = self.edges[y:y+h, x:x+w]

            size = max(h, w)
            square = np.zeros((size, size), np.uint8)

            x = (size - w) // 2
            y = (size - h) // 2

            square[y:y+h, x:x+w] = cropped_img

            square = cv2.resize(square, (28, 28))
            
            input_data = np.expand_dims(square, axis=-1)
            input_data = np.expand_dims(square, axis=0)
            predictions = self.model.predict(input_data, verbose=0)
            print(np.argmax(predictions), end="")

    def __del__(self):
        cv2.destroyAllWindows()


# Пример использования класса
if __name__ == "__main__":
    try:
        paths = ("path/to/image")
        dict = {}
        for i in paths:
            dict[i] = Detection(i)
            dict[i].detections()
            dict[i].show_images()
            print()
    except Exception as e:
        print(f"Произошла ошибка: {e}")