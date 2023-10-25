# please run "pip install scikit-learn" before running this code

import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import fetch_olivetti_faces
import math

class EigenFace:
    def __init__(self, img_matrix, num_imgs, img_w, img_h, img_names):
        self.num_imgs = num_imgs
        self.img_w = img_w
        self.img_h = img_h
        self.img_matrix = self._reshape_img_matrix(img_matrix, num_imgs, img_w, img_h)
        self.mean_vec = self._calc_mean_vec()
        self.norm_img_matrix = self._calc_norm_img_matrix()
        self.names = img_names
        self.eigenfaces = self._get_eigenfaces()

    def _reshape_img_matrix(self, img_matrix, num_imgs, img_w, img_h):
        return np.resize(img_matrix, (num_imgs, img_w * img_h))

    def _calc_mean_vec(self):
        return np.sum(self.img_matrix, axis=0, dtype='float64') / self.num_imgs

    def _calc_norm_img_matrix(self):
        mean_matrix = np.tile(self.mean_vec, (self.num_imgs, 1))
        return self.img_matrix - mean_matrix

    def _get_eigenfaces(self):
        common_eigenvec = (self.norm_img_matrix @ self.norm_img_matrix.T) / self.num_imgs
        eigenvals, eigenvecs = np.linalg.eig(common_eigenvec)
        sorted_idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[sorted_idx]
        eigenvecs = eigenvecs[:, sorted_idx]
        cov_eigenvec = self.norm_img_matrix.T @ eigenvecs
        eigenfaces = preprocessing.normalize(cov_eigenvec.T)
        return eigenfaces
    
    def _plot_imgs(self, images, titles, h, w, rows, cols):
        plt.figure(figsize=(2.2 * cols, 2.2 * rows))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(rows * cols):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i])
            plt.xticks(())
            plt.yticks(())

    def plot_norm_imgs(self, rows, cols):
        self._plot_imgs(self.norm_img_matrix, self.names, self.img_w, self.img_h, rows, cols)

    def plot_eigenfaces(self, rows, cols):
        eigenface_labels = list(range(self.eigenfaces.shape[0]))
        self._plot_imgs(self.eigenfaces, eigenface_labels, self.img_w, self.img_h, rows, cols)

    def recognize_face(self, test_img, num_eigenfaces):
        mean_sub_testimg = np.reshape(test_img, (test_img.shape[0] * test_img.shape[1])) - self.mean_vec
        omega = self.eigenfaces[:num_eigenfaces].dot(mean_sub_testimg)

        min_val, idx = None, None
        for i in range(self.num_imgs):
            omega_i = self.eigenfaces[:num_eigenfaces].dot(self.norm_img_matrix[i])
            diff = omega - omega_i
            similarity = math.sqrt(diff.dot(diff))
            if min_val is None or min_val > similarity:
                min_val, idx = similarity, i

        return (min_val, self.names[idx])


# Fetch the dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
imgs = data.images
names = data.target
num_imgs, img_h, img_w = imgs.shape[0], imgs.shape[1], imgs.shape[2]

# Separate the dataset into training and testing sets
train_imgs = np.array([imgs[i] for i in range(num_imgs) if i % 10 < 9])
test_imgs = np.array([imgs[i] for i in range(num_imgs) if i % 10 == 9])
train_names = [names[i] for i in range(num_imgs) if i % 10 < 9]
test_names = [names[i] for i in range(num_imgs) if i % 10 == 9]

train_matrix = train_imgs.reshape((train_imgs.shape[0], img_h * img_w))

# Initialize EigenFace class
eigenface = EigenFace(train_matrix, train_imgs.shape[0], img_h, img_w, train_names)

# Test face recognition with a test image
test_img = test_imgs[3]
result = eigenface.recognize_face(test_img, 10)

# Display test image and result image
plt.figure(figsize=(8, 4))  # Create a new figure

# Plotting test image
plt.subplot(1, 2, 1)  # First subplot for the test image
plt.imshow(test_img, cmap=plt.cm.gray)
plt.title("Test Image")
plt.axis("off")

# Plotting result image
plt.subplot(1, 2, 2)  # Second subplot for the result image
plt.imshow(train_imgs[result[1]].reshape((img_h, img_w)), cmap=plt.cm.gray)
plt.title(f"Closest Match, Similarity: {result[0]:.2f}")
plt.axis("off")

plt.show()