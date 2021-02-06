import glob
import time

import numpy as np
import tensorflow as tf
from keras import Input
from keras.applications import VGG19
from keras.models import Model

from networks import Generator, Discriminator

def create_vgg(input_shape):
    """
    Build VGG network to extract image features
    """

    # Load a pre-trained VGG19 model trained on 'Imagenet' dataset
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]

    input_layer = Input(shape=input_shape)

    # Extract features
    features = vgg(input_layer)

    # Create a Keras model
    model = Model(inputs=[input_layer], outputs=[features])
    return model

def create_adversarial_model(low_resolution_shape, high_resolution_shape, optimizer):

  # Build and compile VGG19 network to extract features
    vgg = create_vgg(high_resolution_shape)
    vgg.trainable = False
    vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    # Build and compile the discriminator network
    discriminator = Discriminator(high_resolution_shape).create_discriminator()
    discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    # Build the generator network
    generator = Generator(low_resolution_shape).create_generator()

    """
    Build and compile the adversarial model
    """

    # Input layers for high-resolution and low-resolution images
    input_high_resolution = Input(shape=high_resolution_shape)
    input_low_resolution = Input(shape=low_resolution_shape)

    # Generate high-resolution images from low-resolution images
    generated_high_resolution_images = generator(input_low_resolution)

    # Extract feature maps of the generated images
    features = vgg(generated_high_resolution_images)

    # Make the discriminator network as non-trainable
    discriminator.trainable = False

    # Get the probability of generated high-resolution images
    probs = discriminator(generated_high_resolution_images)

    # Create and compile an adversarial model
    adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
    adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=optimizer)
    
    return discriminator, generator, adversarial_model, vgg
