import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras_preprocessing.image import img_to_array, load_img
from scipy.misc import imsave, imread
from skimage.transform import resize

def normalize(input_images_arr):
  input_images_arr = input_images_arr / 127.5 - 1.
  return input_images_arr

def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    # Make a list of all images inside the data directory
    all_images = glob.glob(data_dir)

    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)

    low_resolution_images = []
    high_resolution_images = []

    for img in images_batch:
        # Get an ndarray of the current image
        img1 = imread(img)
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_high_resolution = resize(img1, high_resolution_shape)
        img1_low_resolution = resize(img1, low_resolution_shape)

        # Do a random horizontal flip
        if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)

        # Normalize images
        img1_high_resolution = normalize(img1_high_resolution)
        img1_low_resolution = normalize(img1_low_resolution)

        high_resolution_images.append(img1_high_resolution)
        low_resolution_images.append(img1_low_resolution)

    # Convert the lists to Numpy NDArrays
    return np.array(high_resolution_images), np.array(low_resolution_images)

def save_images(low_resolution_image, original_image, generated_image, path):
  """
  Save low-resolution, high-resolution(original) and
  generated high-resolution images in a single image
  """
  fig = plt.figure()
  ax = fig.add_subplot(1, 3, 1)
  ax.imshow(low_resolution_image)
  ax.axis("off")
  ax.set_title("Low-resolution")

  ax = fig.add_subplot(1, 3, 2)
  ax.imshow(original_image)
  ax.axis("off")
  ax.set_title("Original")

  ax = fig.add_subplot(1, 3, 3)
  ax.imshow(generated_image)
  ax.axis("off")
  ax.set_title("Generated")

  plt.savefig(path)

def save_weights(generator, discriminator, networks_dir):
  # Save models
  generator.save(networks_dir+"generator.h5")
  discriminator.save(networks_dir+"discriminator.h5")

