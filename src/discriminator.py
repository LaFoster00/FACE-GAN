from keras import models, layers, models, ops
import keras
from layers import WeightedAdd



def conv_block(
        x,
        filters,
        activation=layers.LeakyReLU(negative_slope=0.2),
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bn=False,
        use_dropout=False,
        drop_value=0.5,
):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    return x


def get_discriminator_model(
        num_color_channels=3,
        start_filters=64,
        max_filters=512,
        target_resolution=256,
):
    """
    Create a progressive discriminator model with multiple resolution scales
    :return:
        List of discriminator models at different scales
    """
    num_blocks = ops.log2(target_resolution / 4)

    discriminator_models = []

    #Start with 4x4 resolution discriminator
    img_input = layers.Input(shape=(4, 4, num_color_channels))
    x = conv_block(img_input, start_filters)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='sigmoid')(x)

    discriminator_models.append(models.Model(img_input, x, name='discriminator_4x4'))

    #Progessive growth of discriminator
    current_filters = start_filters
    current_resolution = 4

    # Create the next downscale steps
    for i in range(num_blocks):
        current_resolution *= 2
        current_filters = min(current_filters * 2, max_filters)

        # New higher resolution input
        img_input = layers.Input(shape=(current_resolution, current_resolution, num_color_channels))
        #Downsample input
        x = layers.AveragePooling2D()(img_input)
        x = conv_block(x, current_filters // 2)
        x = conv_block(x, current_filters, strides=(2, 2))
        x = conv_block(x, current_filters)



