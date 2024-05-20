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

        self.fill_b_mask(b_mask, masks)
        expanded_b_mask = self.fill_expanded_b_mask(b_mask)

        vectorized_pixels, map_xy_to_position = self.prepare_vectorized_pixels(img, expanded_b_mask)
        self.log(3, "vectorized_pixels len: {0}".format(len(vectorized_pixels)))
        label, K = self.optimize_k_means(vectorized_pixels)

        label_img = self.create_label_img(img.shape[:2], label, map_xy_to_position)
        self.log(3, "label_img shape: {0}".format(label_img.shape))

        hairy_label = self.find_hairy_label(b_mask, K, label_img)
        self.log(3, "hairy_label: {0}".format(hairy_label))

        self.remove_not_hairy_from_b_mask(b_mask, label_img, hairy_label)

        return (img, b_mask)

    def find_contours(self, original_image_path):
        # Run batched inference on a list of images
        original_image = cv.imread(original_image_path)
        if not original_image:
            return original_image, None
        result = self.model(original_image).pop()  # return a list of Results objects

        masks = result.masks

        img = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        return (img, masks)

    def fill_b_mask(self, b_mask, masks):
        for contour in masks.xy:
            contour = contour.astype(np.int32)
            contour = contour.reshape(-1, 1, 2) # make it into a shape that drawContours prefers
            _ = cv.drawContours(b_mask, [contour], -1, (255, 255, 255), cv.FILLED)

    def fill_expanded_b_mask(self, b_mask):
        area = 0
        for y, b_mask_y in enumerate(b_mask):
            for x, mask in enumerate(b_mask_y):
                if mask:
                    area += 1

        ## apply convolution to expand the mask area to give room for non-hair colors to dominate
        expand_factor = int(np.ceil((np.sqrt(2) - 1.0) * np.sqrt(area)))
        # have a circle kernel
        self.log(3, "area: {0}".format(area))
        self.log(3, "expand_factor: {0}".format(expand_factor))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(expand_factor, expand_factor))
        expanded_b_mask = cv.filter2D(b_mask, -1, kernel)

        return expanded_b_mask

    def prepare_vectorized_pixels(self, img, expanded_b_mask):
        map_xy_to_position = {}
        vectorized_pixels = []
        for y, mask_y in enumerate(expanded_b_mask):
            for x, mask in enumerate(mask_y):
                if mask:
                    map_xy_to_position[(x, y)] = len(vectorized_pixels)
                    vectorized_pixels.append(img[y][x])

        vectorized_pixels = np.array(vectorized_pixels)
        vectorized_pixels = np.float32(vectorized_pixels)

        return (vectorized_pixels, map_xy_to_position)

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

    def create_label_img(self, img_shape, label, map_xy_to_position):
        label_img = np.full(img_shape, -1, np.uint8)
        for y in range(label_img.shape[0]):
            for x in range(label_img.shape[1]):
                if (x, y) in map_xy_to_position:
                    position = map_xy_to_position[(x, y)]
                    label_img[y][x] = label[position][0]
        return label_img

    def find_hairy_label(self, b_mask, K, label_img):
        count_hair_label = [0] * K
        count_not_hair_label = [0] * K
        for y, l_y in enumerate(label_img):
            for x, l in enumerate(l_y):
                if l != 255:
                    if b_mask[y][x]:
                        count_hair_label[l] += 1
                    else:
                        count_not_hair_label[l] += 1

        self.log(3, "count_hair_label: {0}".format(count_hair_label))
        self.log(3, "count_not_hair_label: {0}".format(count_not_hair_label))
        hairy_label = set()

        for i in range(K):
            if count_hair_label[i] > count_not_hair_label[i]:
                hairy_label.add(i)
        return hairy_label

    def remove_not_hairy_from_b_mask(self, b_mask, label_img, hairy_label):
        """
            Modify b_mask to only contains hairy pixels
        """
        for y, l_y in enumerate(label_img):
            for x, l in enumerate(l_y):
                if l != 255 and l not in hairy_label and b_mask[y][x]:
                    b_mask[y][x] = 0

    def log(self, level, s):
        if level > 1:
            print(s)

## Helper function to predict
def predict(image_path):
    hair_segment_predictor = HairSegmentPredictor()
    hair_segment_predictor.setup()

    img, b_mask = hair_segment_predictor.find_mask(image_path)

    for y in range(b_mask.shape[0]):
      for x in range(b_mask.shape[1]):
        if b_mask[y][x]:
          img[y][x] = [255,255,255]
    return img


## Helper function to predict and show
def predict_and_show(image_path):
    img = predict(image_path)

    from PIL import Image as Img
    Img.fromarray(img).show()

def make_hsv_dataset(input_dir, output_dir):
    hair_segment_predictor = HairSegmentPredictor()
    hair_segment_predictor.setup()

    import glob
    import json
    for image_path in glob.glob(os.path.abspath(input_dir) + '/*.jpg'):
        hair_segment_predictor.log(4, "image_path: {0}".format(image_path))
        image_name = os.path.basename(image_path)
        data_output_path = os.path.join(output_dir, image_name)
        hair_segment_predictor.log(4, "data_output_path: {0}".format(data_output_path))
        if os.path.exists(data_output_path):
            continue
        data = make_hsv_data(image_path, output_dir, hair_segment_predictor)
        if not data:
            continue
        hair_segment_predictor.log(4, "data: {0}".format(data))
        with open(data_output_path, 'w') as f:
            json.dump(data, f)

def make_hsv_data(image_path, output_dir, hair_segment_predictor = None):
    if not hair_segment_predictor:
        hair_segment_predictor = HairSegmentPredictor()
        hair_segment_predictor.setup()

    img, b_mask = hair_segment_predictor.find_mask(image_path)
    if not b_mask:
        return None

    sample = []
    for y in range(b_mask.shape[0]):
      for x in range(b_mask.shape[1]):
        if b_mask[y][x]:
          sample.append(img[x][y])

    sample_in_hue = cv.cvtColor(np.array([sample]), cv.COLOR_RGB2HSV)[0][:, :1]
    sample_in_hue = sample_in_hue.reshape((sample_in_hue.shape[0],))
    hair_segment_predictor.log(3, "sample shape: {0}, sample type: {1}".format(sample_in_hue.shape, sample_in_hue.dtype))
    min_hue = 255
    max_hue = 0
    mean_hue = 0.0
    for h in sample_in_hue:
        min_hue = min(min_hue, h)
        max_hue = max(max_hue, h)
        mean_hue += h

    mean_hue /= np.float32(len(sample_in_hue))

    median_hue = 0
    if len(sample_in_hue) % 2 == 0:
        middle = len(sample_in_hue) // 2
        median_hue = (sample_in_hue[middle] + sample_in_hue[middle + 1]) / 2
    else:
        median_hue = sample_in_hue[len(sample_in_hue) // 2 + 1]

    data = {
        'min_hue': min_hue.item(),
        'max_hue': max_hue.item(),
        'mean_hue': mean_hue.item(),
        'median_hue': median_hue.item()
    }
    return data

