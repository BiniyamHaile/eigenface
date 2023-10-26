import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import fetch_olivetti_faces
import math

class FisherFace:
    def __init__(self, img_matrix, num_imgs, img_w, img_h, img_names):
        self.num_imgs = num_imgs
        self.img_w = img_w
        self.img_h = img_h
        self.img_matrix = self._reshape_img_matrix(img_matrix, num_imgs, img_w, img_h)
        self.names = img_names
        self.fisherfaces = self._get_fisherfaces()

    def _reshape_img_matrix(self, img_matrix, num_imgs, img_w, img_h):
        return np.resize(img_matrix, (num_imgs, img_w * img_h))

    def _get_fisherfaces(self):
        lda = LDA(n_components=None)
        lda.fit(self.img_matrix, self.names)
        return lda.coef_

    def recognize_face(self, test_img):
        test_img = np.reshape(test_img, (test_img.shape[0] * test_img.shape[1]))
        lda = LDA(n_components=None)
        lda.fit(self.img_matrix, self.names)
        return lda.predict(test_img.reshape(1, -1))

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

# Initialize FisherFace class
fisherface = FisherFace(train_matrix, train_imgs.shape[0], img_h, img_w, train_names)

# Test face recognition with a test image
test_img = test_imgs[3]
result = fisherface.recognize_face(test_img)

# Display test image and result image
plt.figure(figsize=(8, 4))

# Plotting test image
plt.subplot(1, 2, 1)
plt.imshow(test_img, cmap=plt.cm.gray)
plt.title("Test Image")
plt.axis("off")

# Plotting result image
plt.subplot(1, 2, 2)
plt.imshow(train_imgs[result[0]].reshape((img_h, img_w)), cmap=plt.cm.gray)
plt.title(f"Closest Match: {result[0]}")
plt.axis("off")

plt.show()
