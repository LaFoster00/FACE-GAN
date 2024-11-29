from keras import layers, models, ops, backend, utils
import keras
import numpy as np

from layers import WeightedAdd, PixelNorm, ToRGB
from src.utils import lerp_clip, cset, lerp

from utils import number_features


# add a generator block
def add_generator_block(old_model):
    # get the end of the last block
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    upsampling = layers.UpSampling2D()(block_end)
    g = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(upsampling)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    g = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(g)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    # add new output layer
    out_image = layers.Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(g)
    # define model
    model1 = models.Model(old_model.input, out_image)
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedAdd()([out_image2, out_image])
    # define model
    model2 = models.Model(old_model.input, merged)
    return [model1, model2]


# define generator models
def define_generator(latent_dim, n_blocks, in_dim=4):
    model_list = list()
    # base model latent input
    in_latent = layers.Input(shape=(latent_dim,))
    # linear scale up to activation maps
    g = layers.Dense(128 * in_dim * in_dim, kernel_initializer='he_normal')(in_latent)
    g = layers.Reshape((in_dim, in_dim, 128))(g)
    # conv 4x4, input block
    g = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    # conv 3x3
    g = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    # conv 1x1, output block
    out_image = layers.Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(g)
    # define model
    model = keras.models.Model(in_latent, out_image)
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_generator_block(old_model)
        # store model
        model_list.append(models)
    return model_list


if __name__ == '__main__':
    # define models
    generators = define_generator(100, 3)
    # spot check
    m = generators[2][1]
    m.summary()
    utils.plot_model(m, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
