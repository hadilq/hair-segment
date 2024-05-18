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
        img, hsv_img, masks = self.find_contours(original_image_path)
        b_mask = np.zeros(img.shape[:2], np.uint8)
        if not masks:
            return (img, b_mask)

        for contour in masks.xy:
            contour = contour.astype(np.int32)
            contour = contour.reshape(-1, 1, 2) # make it into a shape that drawContours prefers
            contour_b_mask = np.zeros(img.shape[:2], np.uint8)
            _ = cv.drawContours(contour_b_mask, [contour], -1, (255, 255, 255), cv.FILLED)
            hair_color, hair_label, label, vectorized_pixels, map_xy_to_index = self.find_hair_color(hsv_img, contour_b_mask)
            label, center = self.optimize_k_means(vectorized_pixels, hair_color)
            self.log(3, "center : {0}, hair_color: {1}".format(center, hair_color))
            label, hair_label, hair_palette = self.make_hair_palette(label, center, hair_color)
            hair_colors_range = self.make_hair_colors_range(hsv_img, contour_b_mask, map_xy_to_index, label, hair_label)
            self.log(3, "hair_palette : {0}".format(hair_palette))
            range_mask = self.make_range_base_mask(hsv_img, hair_palette, hair_color, hair_colors_range)


            for y, mask_y in enumerate(contour_b_mask):
                for x, mask in enumerate(mask_y):
                    if mask and (x, y) in map_xy_to_index:
                        index = map_xy_to_index[(x, y)]
                        l = label[index][0]
                        if l == hair_label:
                            self.log(1, "label mask: {0}, x: {1}, y: {1}".format(mask, x, y))
                            b_mask[y][x] = 255
                        if range_mask[y][x]:
                            self.log(1, "range mask: {0}, x: {1}, y: {1}".format(mask, x, y))
                            b_mask[y][x] = 255

            b_mask = cv.bitwise_or(b_mask, self.follow_hairs(img, contour_b_mask, hair_palette))
            b_mask = self.fill_holes(hsv_img, b_mask)

        return (img, b_mask)

    def find_contours(self, original_image_path):
        # Run batched inference on a list of images
        result = self.model(original_image_path).pop()  # return a list of Results objects

        masks = result.masks

        original_image = cv.imread(original_image_path)
        img = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        hsv_img = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
        return (img, hsv_img, masks)

    def find_hair_color(self, img, b_mask):
        map_xy_to_index = {}
        vectorized = []

        for y, mask_y in enumerate(b_mask):
            for x, mask in enumerate(mask_y):
                if mask:
                    map_xy_to_index[(x, y)] = len(vectorized)
                    vectorized.append(img[y][x])

        vectorized = np.array(vectorized)
        vectorized = np.float32(vectorized)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        K = 2
        attempts = 10
        _, label, center = cv.kmeans(vectorized, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)

        zeros = 0
        for l in label:
            if l[0] == 0:
                zeros += 1

        ones = 0
        for l in label:
            if l[0] == 1:
                ones+= 1

        if zeros > ones:
            majority = 0
        else:
            majority = 1
        return center[majority], majority, label, vectorized, map_xy_to_index

    def optimize_k_means(self, vectorized, hair_color):
        if len(vectorized) < 2:
            self.log(3, "vectorized's len is {0}".format(vectorized))
            return [0] * len(vectorized), [hair_color] * len(vectorized)

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
        for K in range(3, min(50, len(vectorized))):
            compactness, label, center = cv.kmeans(vectorized, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
            silhouette_score = silhouette_coefficient(label, center)
            if silhouette_score > prev_silhouette_score:
                # reached optimal K
                return label, center

            prev_silhouette_score = silhouette_score
        return label, center

    def make_hair_palette(self, label, center, hair_color):
        """
        label is the list of labels.
        center is the list of colors of each label. Color in HSV.
        hair_color is the color of hair in HSV.
        """
        if len(label) <= 2:
            self.log(3, "label's len is {0}".format(label))
            return label, 0 # FIXME

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = 10

        # by putting hair_av_color in the center, clustering centers to hair and not hair.
        polar = [[]] * len(center)
        h_car = self.hsv_to_cart(hair_color[0], hair_color[1], hair_color[2])
        for i, c in enumerate(center):
            c_car = self.hsv_to_cart(c[0], c[1], c[2])
            p = self.cart2pol_coordinate(c_car[0] - h_car[0], c_car[1] - h_car[1], c_car[2] - h_car[2])
            polar[i] = np.array(p)
        polar = np.array(polar)
        polar = np.float32(polar)

        _, polar_label, polar_center = cv.kmeans(polar, 2, None, criteria, attempts, cv.KMEANS_PP_CENTERS)

        if  polar_center[0][0] < polar_center[1][0]:
            hair_label = 0
        else:
            hair_label = 1

        binary_label = [0] * len(label)
        for i, l in enumerate(label):
            binary_label[i] = polar_label[l[0]]

        hair_palette = []
        for i, c in enumerate(center):
            if hair_label == polar_label[i][0]:
                hair_palette.append(c)

        binary_label = np.array(binary_label)
        self.log(1, "polar: {0}, polar_label: {1}, binary_label: {2}".format(polar, polar_label, binary_label))

        return binary_label, hair_label, hair_palette

    def make_hair_colors_range(self, hsv_img, contour_b_mask, map_xy_to_index, label, hair_label):
        hair_colors_range = []
        for y, mask_y in enumerate(contour_b_mask):
            for x, mask in enumerate(mask_y):
                if mask and (x, y) in map_xy_to_index:
                    index = map_xy_to_index[(x, y)]
                    l = label[index][0]
                    if l == hair_label:
                        hair_colors_range.append(np.float32(hsv_img[y][x]))
        return hair_colors_range


    def make_range_base_mask(self, hsv_img, hair_palette, hair_color, hair_colors_range):
        b_mask = np.zeros(hsv_img.shape[:2], np.uint8)
        for c in hair_colors_range:
            lower = np.array([c[0], 0, 0])
            upper = np.array([hair_color[0], 255, 255])
            self.log(1, "lower: {0}, upper: {1}".format(lower.dtype, upper.dtype))
            mask = cv.inRange(hsv_img, lower, upper)
            b_mask = cv.bitwise_or(b_mask, mask)
        return b_mask

    def follow_hairs(self, img, b_mask, hair_palette):
        return self.follow_hairs_2(img, b_mask, hair_palette)

    def follow_hairs_1(self, img, b_mask, hair_palette):
        epsilon = 40.0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)]
        visited = np.zeros(img.shape[:2] + (len(hair_palette),), np.bool_)
        result_b_mask = np.zeros(img.shape[:2], np.uint8)

        def bfs_walk(x, y, c, ci):
            stack = [(x, y)]
            while stack:
                head = stack.pop()
                for d in directions:
                    nx = head[0] + d[0]
                    ny = head[1] + d[1]
                    if nx < 0 or nx >= img.shape[1] or ny < 0 or ny >= img.shape[0]:
                        continue
                    if (not b_mask[ny][nx]) and (not visited[ny][nx][ci])\
                        and cv.norm(img[ny][nx] - c) < epsilon:
                        visited[ny][nx][ci] = True
                        result_b_mask[ny][nx]= 255
                        stack.append((nx, ny))

        for y, mask_y in enumerate(b_mask):
            for x, mask in enumerate(mask_y):
                if mask:
                    for ci, c in enumerate(hair_palette):
                        if (not visited[y][x][ci]) and cv.norm(img[y][x] - c) < epsilon:
                            visited[y][x][ci] = True
                            bfs_walk(x, y, c, ci)
                            break
        return result_b_mask

    def follow_hairs_2(self, img, b_mask, hair_palette):
        epsilon = 0.5
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)]
        visited = np.zeros(img.shape[:2], np.bool_)
        result_b_mask = np.zeros(img.shape[:2], np.uint8)

        def bfs_walk(x, y, c):
            stack = [(x, y, c)]
            while stack:
                head = stack.pop()
                for d in directions:
                    nx = head[0] + d[0]
                    ny = head[1] + d[1]
                    if nx < 0 or nx >= img.shape[1] or ny < 0 or ny >= img.shape[0]:
                        continue
                    if (not b_mask[ny][nx]) and (not visited[ny][nx])\
                        and cv.norm(img[ny][nx] - head[2]) < epsilon:
                        visited[ny][nx] = True
                        result_b_mask[ny][nx]= 255
                        stack.append((nx, ny, img[ny][nx]))

        for y, mask_y in enumerate(b_mask):
            for x, mask in enumerate(mask_y):
                if mask:
                    if (not visited[y][x]):
                        visited[y][x] = True
                        bfs_walk(x, y, img[y][x])

        return result_b_mask

    def fill_holes(self, img, b_mask):
        img_floodfill = b_mask.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv.floodFill(img_floodfill, mask, (0,0), 255);
        img_floodfill_inv = cv.bitwise_not(img_floodfill)
        return b_mask | img_floodfill_inv

    def hsv_to_cart(self, H, S, V):
        """
        we need this kind of transfer due to the fact that points close to H = 0 and H = 360,
        in HSV(OpenCV version), are adjacent.
        """
        return self.cyl2cart_coordinate(S, H * 2 * np.pi / 360, V)

    def cart_to_hsv(self, x, y, z):
        """ we need this kind of transfer due to the fact that H = 0 and H = 360,
            in HSV(OpenCV version), are adjacent.
        """
        cyl = self.cart2cyl_coordinate(x, y, z)
        return np.array([cyl[0], cyl[1] * 360 / (2 * np.pi), cyl[2]], np.int8)

    def cart2cyl_coordinate(self, x, y, z):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return [rho, phi, z]

    def cyl2cart_coordinate(self, rho, phi, z):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return [x, y, z]

    def cart2pol_coordinate(self, x, y, z):
        rho = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        return [rho, phi, theta]

    def pol2cart_coordinate(self, rho, phi, theta):
        x = rho * np.cos(phi) * np.sin(theta)
        y = rho * np.sin(phi) * np.sin(theta)
        z = rho * np.cos(theta)
        return [x, y, z]

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

