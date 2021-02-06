import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import img_to_array, load_img
from scipy.misc import imsave, imread
from skimage.transform import resize

import utils
from gan import create_adversarial_model, create_vgg
from utils import sample_images, save_images, save_weights

def train(data_dir , img_results_dir, networks_dir , loss_file_dir , epochs, batch_size ):
    
    optimizer = Adam(0.0002, 0.5)

    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    discriminator, generator, adversarial_model, vgg = create_adversarial_model(low_resolution_shape, high_resolution_shape, optimizer)

    for epoch in range(epochs):
      print("Epoch:{}".format(epoch))

      """
      Train the discriminator network
      """

      # Sample a batch of images
      high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                    low_resolution_shape=low_resolution_shape,
                                                                    high_resolution_shape=high_resolution_shape)

      # Generate high-resolution images from low-resolution images
      generated_high_resolution_images = generator.predict(low_resolution_images)

      # Generate batch of real and fake labels
      real_labels = np.ones((batch_size, 1))
      fake_labels = np.zeros((batch_size, 1))

      # Train the discriminator network on real and fake images
      d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
      d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)

      # Calculate total discriminator loss
      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
      #print("d_loss:", d_loss)

      """
      Train the generator network
      """

      # Sample a batch of images
      high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                    low_resolution_shape=low_resolution_shape,
                                                                    high_resolution_shape=high_resolution_shape)

      # Extract feature maps for real high-resolution images
      image_features = vgg.predict(high_resolution_images)

      # Train the generator network
      g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],
                                        [real_labels, image_features])

      #print("g_loss:", str(g_loss))

      loss_file = open(loss_file_dir, 'a')
      loss_file.write('epoch %d : gan_loss = %s ; discriminator_loss_1 = %f; discriminator_loss_2 = %f\n' %(epoch, str(g_loss), d_loss[0], d_loss[1]))
      loss_file.close()

      # Sample and save images after every 100 epochs
      if epoch % 100 == 0:
          high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                        low_resolution_shape=low_resolution_shape,
                                                                        high_resolution_shape=high_resolution_shape)

          generated_images = generator.predict_on_batch(low_resolution_images)

          for index, img in enumerate(generated_images):
              save_images(low_resolution_images[index], high_resolution_images[index], img,
                          path=(img_results_dir+'img_{}_{}').format(epoch, index))
      if epoch % 1000 == 0:
          save_weights(generator, discriminator, networks_dir)

import argparse

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_dir', action='store', dest='data_dir', default='/content/img_align_celeba/*.jpg' ,
                    help='Path for input images')
                    
    parser.add_argument('-o', '--output_dir', action='store', dest='img_results_dir', default='/content/drive/MyDrive/ColabNotebooks/SRGAN/results/' ,
                    help='Path for result images')
    
    parser.add_argument('-m', '--model_save_dir', action='store', dest='networks_dir', default='/content/drive/MyDrive/ColabNotebooks/SRGAN/models/' ,
                    help='Path for networks: generator and discriminator')

    parser.add_argument('-l', '--loss_dir', action='store', dest='loss_file_dir', default='/content/drive/MyDrive/ColabNotebooks/SRGAN/loss.txt' ,
                    help='Path for loss values per epoch')                

    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=1,
                    help='Batch Size', type=int)
                    
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=30000 ,
                    help='Number of iteratios for trainig', type=int)
                    
    
    values = parser.parse_args()
    
    train(values.data_dir, values.img_results_dir, values.networks_dir, values.loss_file_dir, values.epochs, values.batch_size)

