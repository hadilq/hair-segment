from typing import List, Dict, Tuple
import numpy as np
import os
from ultralytics import YOLO
import cv2 as cv

class HairSegmentPredictor:
    def setup(self):
        self.model = YOLO('best.pt')

    def find_mask(self, original_image_path):
        img, masks = self.find_contours(original_image_path)
        b_mask = np.zeros(img.shape[:2], np.uint8)
        if not masks:
            return (img, b_mask)

        for contour in masks.xy:
            contour = contour.astype(np.int32)
            contour = contour.reshape(-1, 1, 2) # make it into a shape that drawContours prefers
            contour_b_mask = np.zeros(img.shape[:2], np.uint8)
            # if you want to uncomment this line, then comment out all the below lines in this loop
            # _ = cv.drawContours(b_mask, [contour], -1, (255, 255, 255), cv.FILLED)
            _ = cv.drawContours(contour_b_mask, [contour], -1, (255, 255, 255), cv.FILLED)
            hair_color, hair_label, label, vectorized_pixels, map_xy_to_index = self.find_hair_color(img, contour_b_mask)
            # You can comment out 3 lines below to compare their effect, without breaking the code!
            label, center = self.optimize_k_means(vectorized_pixels, hair_color)
            self.log(2, "center : {0}, hair_color: {1}".format(center, hair_color))
            label, hair_label, hair_palette = self.merge_back_segments(label, center, hair_color)

            for y, mask_y in enumerate(contour_b_mask):
                for x, mask in enumerate(mask_y):
                    if mask and (x, y) in map_xy_to_index:
                        index = map_xy_to_index[(x, y)]
                        l = label[index][0]
                        if l == hair_label:
                            self.log(1, "mask: {0}, x: {1}, y: {1}".format(mask, x, y))
                            b_mask[y][x] = 255
            _ = self.follow_hairs(img, b_mask, hair_palette)

        return (img, b_mask)

    def find_contours(self, original_image_path):
        # Run batched inference on a list of images
        result = self.model(original_image_path).pop()  # return a list of Results objects

        masks = result.masks

        original_image = cv.imread(original_image_path)
        img = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        return (img, masks)

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
            a = 0
            b = 0
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
        for K in range(2, min(50, len(vectorized))):
            compactness, label, center = cv.kmeans(vectorized, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
            silhouette_score = silhouette_coefficient(label, center)
            if silhouette_score > prev_silhouette_score:
                # reached optimal K
                return label, center

            prev_silhouette_score = silhouette_score
        return label, center

    def merge_back_segments(self, label, center, hair_color):
        if len(label) <= 2:
            self.log(3, "label's len is {0}".format(label))
            return label, 0 # FIXME

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = 10

        # by putting hair_av_color in the center, clustering centers to hair and not hair.
        polar = [[]] * len(center)
        for i, c in enumerate(center):
            p = self.cart2pol_coordinate(c[0] - hair_color[0], c[1] - hair_color[1], c[2] - hair_color[2])
            polar[i] = np.array(p)
        polar = np.array(polar)
        polar = np.float32(polar)

        _, polar_label, polar_center = cv.kmeans(polar, 2, None, criteria, attempts, cv.KMEANS_PP_CENTERS)

        if  polar_center[0][0] < polar_center[1][0]:
            hair_label = 0
        else:
            hair_label = 1

        binary_label = [0] * len(label)
        hair_center = []
        for i, l in enumerate(label):
            binary_label[i] = polar_label[l[0]]
            if binary_label[i] == hair_label:
                hair_center.append(center[l[0]])

        binary_label = np.array(binary_label)
        self.log(1, "polar: {0}, polar_label: {1}, binary_label: {2}".format(polar, polar_label, binary_label))

        return binary_label, hair_label, hair_center

    def follow_hairs(self, img, b_mask, hair_palette):
        self.follow_hairs_2(img, b_mask, hair_palette)

    def follow_hairs_1(img, b_mask, hair_palette):
        epsilon = 10.0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)]
        visited = np.zeros(img.shape[:2] + (len(hair_palette),), np.bool_)
        stack = []

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
                        b_mask[ny][nx]= 255
                        stack.append((nx, ny))

        for y, mask_y in enumerate(b_mask):
            for x, mask in enumerate(mask_y):
                if mask:
                    for ci, c in enumerate(hair_palette):
                        if (not visited[y][x][ci]) and cv.norm(img[y][x] - c) < epsilon:
                            b_mask[y][x] = 255
                            visited[y][x][ci] = True
                            bfs_walk(x, y, c, ci)
                            break
        return b_mask

    def follow_hairs_2(self, img, b_mask, hair_palette):
        epsilon = 1.0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)]
        visited = np.zeros(img.shape[:2], np.bool_)
        stack = []

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
                        b_mask[ny][nx]= 255
                        stack.append((nx, ny, img[ny][nx]))

        for y, mask_y in enumerate(b_mask):
            for x, mask in enumerate(mask_y):
                if mask:
                    if (not visited[y][x]):
                        visited[y][x] = True
                        b_mask[y][x] = 255
                        bfs_walk(x, y, img[y][x])
        return b_mask

    def cart2pol_coordinate(self, x, y, z):
        rho = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        return [rho, phi, theta]

    def log(self, level, s):
        if level > 1:
            print(s)

