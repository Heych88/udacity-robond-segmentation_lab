import os
import glob
import sys
import tensorflow as tf

from scipy import misc
import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from tensorflow import image

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools
from utils import model_tools

learning_rate = 0.01
batch_size = 128
num_epochs = 1
steps_per_epoch = 500
validation_steps = 50
workers = 20

def separable_conv2d_batchnorm(input_layer, filters, strides=1, kernel=3):
    output_layer = SeparableConv2DKeras(filters=filters, kernel_size=kernel, strides=strides,
                                        padding='same', activation='elu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer


def conv2d_batchnorm(input_layer, filters, kernel=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides,
                                 padding='same', activation='elu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer

def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer

def average_pool(inputs, kernel=2, stride=2, padding='same'):
    ksize = [kernel, kernel]
    strides = [stride, stride]
    return layers.AveragePooling2D(pool_size=ksize, strides=strides, padding=padding)(inputs)

def encoder_block(input_layer, filters):
    x = separable_conv2d_batchnorm(input_layer, filters, strides=1, kernel=1)
    x = separable_conv2d_batchnorm(x, filters, strides=1, kernel=3)
    p1 = average_pool(x)

    r1 = separable_conv2d_batchnorm(input_layer, filters, strides=2, kernel=3)
    return layers.add([r1, p1])

def decoder_block(small_ip_layer, filters):
    x = bilinear_upsample(small_ip_layer)  # 16 x 16
    x = separable_conv2d_batchnorm(x, filters, strides=1, kernel=1)

    #x = separable_conv2d_batchnorm(x, filters, strides=1, kernel=1)
    x = separable_conv2d_batchnorm(x, filters, strides=1, kernel=3)

    return x


def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks.
    filters = 16
    keep_prob = 1
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    l1 = encoder_block(inputs, filters)
    l1 = keras.layers.core.Dropout(keep_prob)(l1)
    l2 = encoder_block(l1, filters*2)
    l2 = keras.layers.core.Dropout(keep_prob)(l2)
    l3 = encoder_block(l2, filters * 3)
    l3 = keras.layers.core.Dropout(keep_prob)(l3)
    l4 = encoder_block(l3, filters * 4)

    fcn = conv2d_batchnorm(l4, filters*4, strides=1, kernel=1)

    # decoder
    x = decoder_block(fcn, filters*4)
    x = decoder_block(x, filters * 3)
    x = keras.layers.core.Dropout(keep_prob)(x)

    x = decoder_block(x, filters * 2)
    x = layers.concatenate([x, l1])
    x = keras.layers.core.Dropout(keep_prob)(x)

    x = decoder_block(x, filters)
    x = layers.concatenate([x, inputs])

    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
image_hw = 128
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(image_shape)
num_classes = 3

# Call fcn_model()
output_layer = fcn_model(inputs, num_classes)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Define the Keras model and compile it for training
model = models.Model(inputs=inputs, outputs=output_layer)

model.compile(optimizer=keras.optimizers.Adam(learning_rate, decay=0.0002), loss='categorical_crossentropy')

# Data iterators for loading the training and validation data
train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                               data_folder=os.path.join('..', 'data', 'train'),
                                               image_shape=image_shape,
                                               shift_aug=True)

val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                             data_folder=os.path.join('..', 'data', 'validation'),
                                             image_shape=image_shape)

logger_cb = plotting_tools.LoggerPlotter()
callbacks = [logger_cb]

model.fit_generator(train_iter,
                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                    epochs = num_epochs, # the number of epochs to train for,
                    validation_data = val_iter, # validation iterator
                    validation_steps = validation_steps, # the number of batches to validate on
                    callbacks=callbacks,
                    workers = workers)

# Save your trained model weights
weight_file_name = 'model_weights'
model_tools.save_network(model, weight_file_name)

# If you need to load a model which you previously trained you can uncomment the codeline that calls the function below.
#weight_file_name = 'model_weights'
#restored_model = model_tools.load_network(weight_file_name)

# generate predictions, save in the runs, directory.
run_number = 'run1'
validation_path, output_path = model_tools.write_predictions_grade_set(model,run_number,'validation')

# take a look at predictions
# validation_path = 'validation'
im_files = plotting_tools.get_im_file_sample(run_number,validation_path)
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)

scoring_utils.score_run(validation_path, output_path)