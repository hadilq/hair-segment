from typing import List, Dict, Tuple
import numpy as np
import os
from ultralytics import YOLO
import cv2 as cv

dir_path = os.path.dirname(os.path.realpath(__file__))

class HairSegmentPredictor:
    def setup(self):
        self.model = YOLO(dir_path + '/best.pt')

    def find_mask(self, original_image_path):
        img, masks = self.find_contours(original_image_path)
        b_mask = np.zeros(img.shape[:2], np.uint8)
        if not masks:
            return (img, b_mask)

        window_top_left_y = img.shape[0]
        window_top_left_x = img.shape[1]
        window_bottom_right_y = 0
        window_bottom_right_x = 0

        for contour in masks.xy:
            contour = contour.astype(np.int32)
            contour = contour.reshape(-1, 1, 2) # make it into a shape that drawContours prefers
            contour_b_mask = np.zeros(img.shape[:2], np.uint8)
            _ = cv.drawContours(b_mask, [contour], -1, (255, 255, 255), cv.FILLED)
            for p in contour:
                x, y = p[0]
                window_top_left_y = min(window_top_left_y, y)
                window_top_left_x = min(window_top_left_x, x)
                window_bottom_right_y = max(window_bottom_right_y, y)
                window_bottom_right_x = max(window_bottom_right_x, x)

        self.log(3, "window: {0}, {1}, {2}, {3}".format(window_top_left_y, window_top_left_x, window_bottom_right_y, window_bottom_right_x))
        window = img[window_top_left_y: window_bottom_right_y + 1, window_top_left_x: window_bottom_right_x + 1][:]

        vectorized_pixels = window.reshape((-1,3))
        vectorized_pixels = np.float32(vectorized_pixels)
        self.log(3, "vectorized_pixels len: {0}".format(len(vectorized_pixels)))
        label, K = self.optimize_k_means(vectorized_pixels)
        label_img = label.flatten().reshape(window.shape[:2])
        self.log(3, "label_img shape: {0}".format(label_img.shape))

        count_hair_label = [0] * K
        count_not_hair_label = [0] * K
        for y, l_y in enumerate(label_img):
            for x, l in enumerate(l_y):
                if b_mask[y + window_top_left_y][x + window_top_left_x]:
                    count_hair_label[l] += 1
                else:
                    count_not_hair_label[l] += 1

        self.log(3, "count_hair_label: {0}".format(count_hair_label))
        self.log(3, "count_not_hair_label: {0}".format(count_not_hair_label))
        hairy_label = set()
        for i in range(K):
            if count_hair_label[i] > count_not_hair_label[i]:
                hairy_label.add(i)
        self.log(3, "hairy_label: {0}".format(hairy_label))

        # Modify b_mask to only contains hairy pixels
        for y, l_y in enumerate(label_img):
            for x, l in enumerate(l_y):
                if l not in hairy_label and b_mask[y + window_top_left_y][x + window_top_left_x]:
                    b_mask[y + window_top_left_y][x + window_top_left_x] = 0

        return (img, b_mask)

    def find_contours(self, original_image_path):
        # Run batched inference on a list of images
        result = self.model(original_image_path).pop()  # return a list of Results objects

        masks = result.masks

        original_image = cv.imread(original_image_path)
        img = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        return (img, masks)

    def optimize_k_means(self, vectorized):
        if len(vectorized) < 2:
            self.log(3, "vectorized's len is {0}".format(vectorized))
            return np.array([0] * len(vectorized)), len(vectorized)

        def silhouette_coefficient(label, center):
            a, b = 0, 0
            for i, p in enumerate(vectorized):
                a += cv.norm(p - center[label[i][0]])
                nearest = float("inf")
                for j, c in enumerate(center):
                    if j != label[i][0]:
                        nearest = min(nearest, cv.norm(p - c))
                b += nearest
            return (b - a) / max(a, b)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = 10

        prev_silhouette_score = float("inf")
        for K in range(3, min(500, len(vectorized))):
            self.log(3, "K: {0}".format(K))
            compactness, label, center = cv.kmeans(vectorized, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
            silhouette_score = silhouette_coefficient(label, center)
            if silhouette_score > prev_silhouette_score:
                # reached optimal K
                return label, K

            prev_silhouette_score = silhouette_score

        return label, K

    def log(self, level, s):
        if level > 1:
            print(s)

## Helper function to predict
def predict(image_path):
    hair_segment_predictor = HairSegmentPredictor()
    hair_segment_predictor.setup()

    img, b_mask = hair_segment_predictor.find_mask(image_path)

    for x in range(len(b_mask)):
      for y in range(len(b_mask[0])):
        if b_mask[x][y]:
          img[x][y] = [255,255,255]
    return img


## Helper function to predict and show
def predict_and_show(image_path):
    img = predict(image_path)

    from PIL import Image as Img
    Img.fromarray(img).show()

