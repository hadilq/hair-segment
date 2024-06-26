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
        if img is None:
            return (img, None)
        b_mask = np.zeros(img.shape[:2], np.uint8)
        if masks is None:
            return (img, b_mask)

        self.fill_b_mask(b_mask, masks)
        area = self.calculate_area(b_mask)
        hairy_label = set()
        while area and not hairy_label:
            expanded_b_mask = self.fill_expanded_b_mask(area, b_mask)

            vectorized_pixels, map_xy_to_position = self.prepare_vectorized_pixels(img, expanded_b_mask)
            log(3, "vectorized_pixels len: {0}", len(vectorized_pixels))
            label, K = self.optimize_k_means(vectorized_pixels)

            label_img = self.create_label_img(img.shape[:2], label, map_xy_to_position)
            log(3, "label_img shape: {0}", label_img.shape)

            hairy_label = self.find_hairy_label(b_mask, K, label_img)
            log(3, "hairy_label: {0}", hairy_label)
            area  *= 2/3

        self.remove_not_hairy_from_b_mask(b_mask, label_img, hairy_label)

        return (img, b_mask)

    def find_contours(self, original_image_path):
        # Run batched inference on a list of images
        original_image = cv.imread(original_image_path)
        if original_image is None or original_image.size == 0:
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

    def calculate_area(self, b_mask):
        area = 0
        for y, b_mask_y in enumerate(b_mask):
            for x, mask in enumerate(b_mask_y):
                if mask:
                    area += 1
        return area

    def fill_expanded_b_mask(self, area, b_mask):
        ## apply convolution to expand the mask area to give room for non-hair colors to dominate
        expand_factor = int(np.ceil((np.sqrt(2) - 1.0) * np.sqrt(area)))
        # have a circle kernel
        log(3, "area: {0}", area)
        log(3, "expand_factor: {0}", expand_factor)
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
            log(3, "vectorized's len is {0}", vectorized)
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
            log(3, "K: {0}", K)
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

        log(3, "count_hair_label: {0}", count_hair_label)
        log(3, "count_not_hair_label: {0}", count_not_hair_label)
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

## Helper function to predict
def predict(image_path):
    hair_segment_predictor = HairSegmentPredictor()
    hair_segment_predictor.setup()

    img, b_mask = hair_segment_predictor.find_mask(image_path)
    if img is None or img.size == 0:
        return None

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
    from PIL import Image as Img
    for image_path in glob.glob(os.path.abspath(input_dir) + '/*.jpg'):
        log(3, "image_path: {0}", image_path)
        image_name = os.path.basename(image_path)
        splitted_name = os.path.splitext(image_name)
        data_name = splitted_name[0] + '.json'
        data_output_path = os.path.join(output_dir, data_name)
        gray_name = splitted_name[0] + '-gray-hair.jpg'
        gray_output_path = os.path.join(output_dir, gray_name)
        b_mask_name = splitted_name[0] + '-b-mask.png'
        b_mask_output_path = os.path.join(output_dir, b_mask_name)
        log(3, "data_output_path: {0}", data_output_path)
        if os.path.exists(data_output_path) and os.path.exists(gray_output_path)\
            and os.path.exists(b_mask_output_path):
            continue
        img, b_mask, data = make_hsv_data(image_path, hair_segment_predictor)
        if data is None:
            continue
        log(3, "data: {0}", data)
        with open(data_output_path, 'w') as f:
            json.dump(data, f)
        gray_img = make_gray_hair(img, b_mask)
        Img.fromarray(gray_img).save(gray_output_path)
        Img.fromarray(b_mask).save(b_mask_output_path)


def make_hsv_data(image_path, hair_segment_predictor = None):
    if not hair_segment_predictor:
        hair_segment_predictor = HairSegmentPredictor()
        hair_segment_predictor.setup()

    img, b_mask = hair_segment_predictor.find_mask(image_path)
    if img is None or img.size == 0:
        return img, b_mask, None

    sample = []
    for y in range(b_mask.shape[0]):
      for x in range(b_mask.shape[1]):
        if b_mask[y][x]:
          sample.append(img[x][y])

    if len(sample) == 0:
        return img, b_mask, None

    sample_in_hue = cv.cvtColor(np.array([sample]), cv.COLOR_RGB2HSV)[0][:, :1]
    sample_in_hue = sample_in_hue.reshape((sample_in_hue.shape[0],))
    log(2, "sample shape: {0}, sample type: {1}", sample_in_hue.shape, sample_in_hue.dtype)
    min_hue = np.uint8(179)
    max_hue = np.uint8(0)
    mean_hue = np.float32(0.0)
    for h in sample_in_hue:
        min_hue = min(min_hue, h)
        max_hue = max(max_hue, h)
        mean_hue += h

    mean_hue /= np.float32(len(sample_in_hue))

    sample_in_hue = np.sort(sample_in_hue, axis=None)
    median_hue = np.uint8(0)
    if len(sample_in_hue) % 2 == 0:
        middle = len(sample_in_hue) // 2
        median_hue = np.uint8((sample_in_hue[middle] + sample_in_hue[middle + 1]) / 2)
    else:
        median_hue = np.uint8(sample_in_hue[len(sample_in_hue) // 2 + 1])

    mean_hue = np.uint8(np.floor(mean_hue))
    data = {
        'min_hue': min_hue.item(),
        'max_hue': max_hue.item(),
        'mean_hue': mean_hue.item(),
        'median_hue': median_hue.item()
    }
    log(3, "data: {0}", data)
    return img, b_mask, data

def make_gray_hair(img, b_mask):
    gray_img = img.copy()
    gray_img = cv.cvtColor(gray_img, cv.COLOR_RGB2GRAY)
    return gray_img

def test_if_all_files_are_parcelable(input_dir):
    import glob
    import json
    for json_path in glob.glob(input_dir+ f'/*.json'):
        with open(json_path, 'r') as f:
            try:
                data = json.loads(f.read())
                log(1, "All good: {0}", json_path)
            except json.JSONDecodeError as e:
                log(2, "Invalid JSON syntax: {0}", e)
                log(2, "json path: {0}", json_path)
                os.remove(json_path)
    for png_path in glob.glob(input_dir+ f'/*.png'):
        b_mask = cv.imread(png_path)
        if b_mask is None:
            log(2, "png path: {0}", png_path)
            os.remove(png_path)


def log(level, s, *arg):
    if level > 2:
       if arg:
           print(s.format(*arg))
       else:
           print(s)

