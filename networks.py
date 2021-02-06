# Modules
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add


# Residual block
def res_block_gen(inp, kernal_size=3, filters=64, strides=1):
        
    res_block = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(inp)
    res_block = BatchNormalization(momentum = 0.5)(res_block)
    # Using Parametric ReLU
    res_block = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(res_block)
    res_block = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(res_block)
    res_block = BatchNormalization(momentum = 0.5)(res_block)
        
    res_block = add([inp, res_block])
    
    return res_block
    
    
def up_sampling_block(inp, filters=256,  kernal_size=3, strides=1):

  up_samp = UpSampling2D(size = 2)(inp)
  up_samp = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(up_samp)
  up_samp = LeakyReLU(alpha = 0.2)(up_samp)
  
  return up_samp


def discriminator_block(inp, filters, kernel_size, strides):
    
    inp = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(inp)
    inp = BatchNormalization(momentum = 0.5)(inp)
    inp = LeakyReLU(alpha = 0.2)(inp)
    
    return inp

class Generator(object):

    def __init__(self, input_shape):
        
        self.input_shape = input_shape
        self.residual_blocks_num = 16

    def create_generator(self):
      gen_input = Input(shape = self.input_shape)
      
      gen1 = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
      gen2 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(gen1)

      res_block = res_block_gen(gen2)
      # Add 16 Residual Blocks
      for index in range(self.residual_blocks_num-1):
        res_block = res_block_gen(res_block)

      gen3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(res_block)
      gen4 = BatchNormalization(momentum = 0.5)(gen3)
      gen5 = add([gen2, gen4])
	    
      # Using 2 UpSampling Blocks
      for index in range(2):
        gen5 = up_sampling_block(gen5)
      gen6 = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(gen5)
      gen7 = Activation('tanh')(gen6)
      
      generator_model = Model(inputs = [gen_input], outputs = [gen7])
      
      return generator_model

class Discriminator(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def create_discriminator(self):
        
        dis_input = Input(shape = self.image_shape)
        
        dis1 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        dis2 = LeakyReLU(alpha = 0.2)(dis1)
        
        dis3 = discriminator_block(dis2, 64, 3, 2)
        dis4 = discriminator_block(dis3, 128, 3, 1)
        dis5 = discriminator_block(dis4, 128, 3, 2)
        dis6 = discriminator_block(dis5, 256, 3, 1)
        dis7 = discriminator_block(dis6, 256, 3, 2)
        dis8 = discriminator_block(dis7, 512, 3, 1)
        dis9 = discriminator_block(dis8, 512, 3, 2)
        
        dis9 = Flatten()(dis9)
        dis10 = Dense(1024)(dis9)
        dis11 = LeakyReLU(alpha = 0.2)(dis10)
       
        dis12 = Dense(1)(dis11)
        dis13 = Activation('sigmoid')(dis12) 
        
        discriminator_model = Model(inputs = [dis_input], outputs = [dis13])
        
        return discriminator_model
